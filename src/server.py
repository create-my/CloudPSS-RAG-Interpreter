"""
CloudPSS Interpreter Server
基于FastAPI的CloudPSS代码生成服务
"""

import sys
from pathlib import Path

# 添加本地 interpreter 目录到 Python 路径（优先于 pip 安装的版本）
# 指向 CloudPSS-RAG-Interpreter/interpreter 目录
LOCAL_INTERPRETER_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LOCAL_INTERPRETER_PATH))

from fastapi import FastAPI
from interpreter import interpreter
import os
import uuid
from typing import Dict, List, Optional, Tuple
import threading
from pathlib import Path

from rag_pipeline import RAGPipeline, SkillPredictor, FailureReasonPredictor

# 获取项目根目录（src的上级目录）
BASE_DIR = Path(__file__).resolve().parent.parent
# 加载.env文件
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="CloudPSS RAG Interpreter",
    description="基于RAG和技能预测的电力系统仿真代码生成服务",
    version="1.0.0"
)

# 全局变量
session_history: Dict[str, List[Dict]] = {}
skill_predictor: Optional[SkillPredictor] = None
skill_predictor_lock = threading.Lock()
conversation_id: Optional[str] = None


def load_skill_predictor():
    """预加载技能预测模型"""
    global skill_predictor
    try:
        with skill_predictor_lock:
            if skill_predictor is None:
                print("正在预加载技能预测模型...")
                # 使用绝对路径
                model_path = str(BASE_DIR / os.getenv("SKILL_MODEL_PATH", "models/failure_reason_predictor.pth"))
                embedding_path = str(BASE_DIR / os.getenv("EMBEDDING_MODEL_PATH", "models/all-MiniLM-L6-v2"))
                skill_predictor = SkillPredictor(
                    model_path=model_path,
                    embedding_model_path=embedding_path
                )
                print("技能预测模型加载完成")
    except Exception as e:
        print(f"警告: 技能预测器初始化失败 - {str(e)}")
        skill_predictor = None


# 应用启动时加载模型
load_skill_predictor()


class OptimizedRAGPipeline(RAGPipeline):
    """优化后的RAGPipeline，使用预加载的技能预测器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global skill_predictor
        if skill_predictor is not None:
            self.skill_predictor = skill_predictor

    def analyze_skills(self, task: str, state: str, documents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """重写技能分析方法"""
        if not documents:
            print("警告: 没有检索到任何文档")
            return [], []

        if not hasattr(self, 'skill_predictor') or self.skill_predictor is None:
            print("错误: 技能预测器不可用")
            return [], []

        try:
            skill_texts = [doc["content"] for doc in documents]
            state_text = task + "\n" + state if state else task + task
            analysis_results = self.skill_predictor.predict_skills(state_text=state_text, skill_texts=skill_texts)

            for doc, analysis in zip(documents, analysis_results):
                doc["analysis"] = analysis

            sorted_docs = sorted(
                documents,
                key=lambda x: x["analysis"]["success_probability"],
                reverse=True
            )

            print("\n技能预测结果（按成功概率排序）:")
            for i, doc in enumerate(sorted_docs[:5], 1):
                analysis = doc["analysis"]
                print(f"{i}. 成功概率: {analysis['success_probability'] * 100:.1f}% - 相似度: {doc['score']:.4f}")

            return sorted_docs, documents

        except Exception as e:
            print(f"技能分析过程中出错: {str(e)}")
            return [], []


def cloudpss_agent(**params):
    """CloudPSS代理函数，处理用户查询并生成响应"""
    # 配置信息
    DATASET_ID = os.getenv("DATASET_ID", "your-dataset-id")
    KB_API_KEY = os.getenv("KB_API_KEY", "your-kb-api-key")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "your-llm-api-key")

    messages = params.get('messages', [])
    if not messages:
        print("警告: 没有收到任何消息")
        return

    global conversation_id
    session_id = conversation_id if conversation_id else str(uuid.uuid4())

    if session_id not in session_history:
        session_history[session_id] = []

    new_messages = [msg for msg in messages[:-1] if msg.get("role") in ("user", "assistant")]
    session_history[session_id].extend(new_messages)
    task = messages[0].get("content", "")
    state = messages[-1].get("content", "")

    if task == state:
        state = ""
    if not state and not task:
        print("警告: 查询内容为空")
        return

    context_messages = str(session_history[session_id] + [{"role": "user", "content": state if state else task}])

    # 使用优化后的RAG管道
    model_path = str(BASE_DIR / os.getenv("SKILL_MODEL_PATH", "models/failure_reason_predictor.pth"))
    embedding_path = str(BASE_DIR / os.getenv("EMBEDDING_MODEL_PATH", "models/all-MiniLM-L6-v2"))
    rag_pipeline = OptimizedRAGPipeline(
        dataset_id=DATASET_ID,
        api_key=KB_API_KEY,
        llm_api_key=LLM_API_KEY,
        model_path=model_path,
        embedding_model_path=embedding_path
    )

    print(f"处理查询: {state if state else task}")
    print("=" * 50)

    full_response = ""
    error_keywords = ["Error", "Traceback", "Exception", "AttributeError", "TypeError",
                      "NameError", "ValueError", "SyntaxError", "KeyError",
                      "IndexError", "ImportError", "ModuleNotFoundError"]

    # 如果非首次查询且没有报错，就不检索技能库
    if len(session_history[session_id]) > 1 and not any(kw in state for kw in error_keywords):
        for chunk in rag_pipeline.only_query_LLM(
                user_query=task + state,
                context_messages=context_messages,
                stream=True
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    full_response += content
                    print(content, end="", flush=True)
                    yield {"choices": [{"delta": {"content": content}}]}
    else:
        for chunk in rag_pipeline.query(
                task=task,
                state=state,
                context_messages=context_messages,
                stream=True
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    full_response += content
                    print(content, end="", flush=True)
                    yield {"choices": [{"delta": {"content": content}}]}

    if full_response:
        session_history[session_id].append({"role": "assistant", "content": full_response})


# CloudPSS配置
cloudpss_api_url = os.environ.get("CLOUDPSS_API_URL", 'https://cloudpss.net/')
cloudpss_Token = os.environ.get('CLOUDPSS_TOKEN', 'your-cloudpss-token')
cloudpss_Model = os.environ.get('CLOUDPSS_MODEL', 'model/username/model-name')

# 初始化interpreter
init_code = f"""
import cloudpss
import os
cloudpss_api_url = '{cloudpss_api_url}'
cloudpss_Token = '{cloudpss_Token}'
cloudpss_Model = '{cloudpss_Model}'
os.environ['CLOUDPSS_API_URL'] = cloudpss_api_url
cloudpss.setToken(cloudpss_Token)
cloudpss_substation_Model = cloudpss.Model.fetch(cloudpss_Model)
"""

try:
    interpreter.computer.run("python", init_code)
except Exception as e:
    print(f"警告: CloudPSS初始化失败 - {e}")

interpreter.auto_run = True
interpreter.llm.model = os.getenv("LLM_MODEL", "gpt-4")
interpreter.llm.api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:9996/v1")
interpreter.llm.api_key = os.getenv("LLM_API_KEY_LOCAL", "anything")
interpreter.max_output = int(os.getenv("MAX_OUTPUT", "2500"))
interpreter.max_message_count = int(os.getenv("MAX_MESSAGE_COUNT", "30"))
interpreter.llm.completions = cloudpss_agent


@app.get("/chat")
def chat_endpoint(message: str):
    """
    聊天端点，处理用户消息

    Args:
        message: 用户输入的消息

    Returns:
        聊天响应
    """
    global conversation_id
    conversation_id = None
    interpreter.llm.completions = cloudpss_agent
    interpreter.messages = []
    the_result_messages = interpreter.chat(message)
    return the_result_messages


@app.get("/history")
def history_endpoint():
    """获取当前会话的历史消息"""
    return interpreter.messages


@app.get("/health")
def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "skill_predictor_loaded": skill_predictor is not None
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))

    print(f"启动服务器: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
