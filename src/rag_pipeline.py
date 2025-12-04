"""
RAG Pipeline for CloudPSS Skill Recommendation
基于向量检索和技能成功率预测的RAG管道
"""

import requests
import json
import os
from typing import List, Dict, Optional, Generator, Tuple
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import re
from pymilvus import connections, Collection
from pymilvus import model as milvus_model


class FailureReasonPredictor(nn.Module):
    """失败原因预测神经网络模型"""

    def __init__(self, input_dim: int, num_classes: int):
        super(FailureReasonPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        logits = self.output_layer(x)
        return logits

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


class SkillPredictor:
    """技能成功率预测器"""

    def __init__(self, model_path: str, embedding_model_path: str):
        """
        初始化技能预测器

        Args:
            model_path: 预测模型路径
            embedding_model_path: 嵌入模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # 加载嵌入模型
            self.embedding_model = SentenceTransformer(embedding_model_path)
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

            # 加载分类模型配置
            checkpoint = torch.load(model_path, map_location=self.device)
            self.reason_to_idx = checkpoint['reason_to_idx']
            self.idx_to_reason = checkpoint['idx_to_reason']
            self.num_classes = len(self.idx_to_reason)

            # 初始化预测模型
            self.model = FailureReasonPredictor(embedding_dim * 2, self.num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # 定义失败原因解释
            self.failure_reason_explanations = {
                "属性错误": """
                1. 对象缺少属性（AttributeError）
常见属性缺失
Component 对象：
缺少 get、label、props、properties、position、name、args、rid、attrs、data 等属性或方法。
示例：'Component' object has no attribute 'label'。
Model 对象：
缺少 getComponents、getModelJobs、getJobResult、emtp、info、plots、getParams、getName、details、getRevisions、attributes、getComponentByLabel、getComponentByName、getImplements、getParameterSchemes、getInfo、get_attributes 等属性或方法。
示例：'Model' object has no attribute 'getComponents'。
其他对象：
Job 对象缺少 get、plots。
FunctionExecution 对象缺少 getPlots。
dict 对象缺少 fs。
根本原因
调用了对象未实现的方法或访问了不存在的属性。
对象类型与预期不符（如误将字符串当作字典或列表）。
2. 对象方法不存在（MethodError）
常见方法缺失
Component 对象缺少 get、getArgs、getAttributes。
Model 对象缺少 addModelJob、submitNewJob、getRevision、getModelComponents、getJob、getElementByLabel、details 等。
其他对象如 Job 缺少 get。
根本原因
调用了未定义的方法，可能是 API 版本不一致或拼写错误。
3. 错误的对象类型操作
字符串（str）误用
尝试调用 str.copy()、str.get() 或访问 str.label、str.key。
示例：'str' object has no attribute 'copy'。
NoneType 误用
尝试调用 None.copy() 或访问 None.label。
示例：'NoneType' object has no attribute 'copy'。
根本原因
变量未正确初始化或赋值，导致实际类型与预期不符（如返回 None 或字符串）。
4. 其他错误
dict 对象缺少 fs：误将字典当作其他对象使用。
list 或 dict 预期错误：如 pins 变量应为列表/字典但实际是字符串或 None。
常见解决方案
检查对象类型：
使用 type(obj) 或 print(obj) 确认对象实际类型。
确保操作（如 .copy()）适用于该类型。
验证属性/方法是否存在：
使用 dir(obj) 查看对象的属性和方法""",
                "变量或引用错误": """NameError，表明代码尝试访问一个未定义或未赋值的变量。
具体错误分析
project_configs 未定义
错误：NameError: name 'project_configs' is not defined
原因：代码中使用了变量 project_configs，但未提前声明或赋值（如缺少 project_configs = {...} 或导入）。
fault_element 未定义
错误：NameError: name 'fault_element' is not defined
原因：变量 fault_element 在被引用前未定义（可能是拼写错误、逻辑分支未覆盖或遗漏赋值）。
model 未定义
错误：NameError: name 'model' is not defined（发生在 vars(model) 调用时）
原因：变量 model 未初始化（如未实例化类、未从模块导入或函数未返回该变量）。
根本原因
变量作用域问题：变量可能定义在局部作用域（如函数内），但在外部被调用。
拼写错误：变量名拼写不一致（如 model vs Model）。
执行顺序问题：代码逻辑中遗漏了对变量的赋值步骤。
依赖缺失：未导入必要的模块或配置（如 project_configs 应从其他文件导入）。
解决方法
检查变量定义：确保变量在使用前已正确赋值（如 model = SomeClass()）""",
                "分类错误": "LLM调用失败",
                "键值或索引错误": """详细的错误总结和调试方法：

1. KeyError: 'label'
错误原因：尝试访问 component.__dict__['label']，但某些元件没有 label 属性。
调试方法：
在访问 label 属性前，先检查 'label' 是否在 component.__dict__ 中。
修改后的代码：
python
if 'label' in component.__dict__:
    label = component.__dict__['label']
2. KeyError: 'component_new_constant_2'
错误原因：尝试通过 rid='component_new_constant_2' 获取元件，但该 rid 不存在。
调试方法：
确认 rid 是否正确，或尝试使用其他已知存在的 rid。
检查模型中是否存在元件：
python
components = cloudpss_substation_Model.getAllComponents()
if components:
    print("存在元件")
else:
    print("没有找到任何元件")
3. KeyError: 0
错误原因：
尝试通过索引 0 访问 components，但 components 可能是一个字典或空列表。
例如：component = components[0] 或 first_job[0]。
调试方法：
确认 components 的类型和内容：
python
print(type(components))  # 检查是列表还是字典
print(components)       # 查看内容
如果是字典，使用 .keys() 或 .values() 访问：
python
if components:
    first_key = list(components.keys())[0]
    component = components[first_key]
4. KeyError: 'rid'
错误原因：尝试访问 j['rid']，但字典 j 中没有 'rid' 键。
调试方法：
检查字典 j 的结构：
python
for j in self.jobs:
    print(j.keys())  # 查看可用键
确保使用正确的键名访问字典。
5. KeyError: 'args'
错误原因：尝试访问 config['args']，但 config 字典中没有 'args' 键。
调试方法：
检查 config 的结构：
python
print(config.keys())  # 查看可用键
如果 'args' 不存在，初始化一个默认值：
python
args = config.get('args', {})  # 如果不存在，返回空字典
6. KeyError: '新的电磁暂态仿真方案'
错误原因：尝试通过 jobType='新的电磁暂态仿真方案' 创建作业，但 JOB_DEFINITIONS 中没有该键。
调试方法：
确认 jobType 的正确值（如 'emtp' 或 'emtps'）：
python
print(JOB_DEFINITIONS.keys())  # 查看支持的作业类型
使用正确的 jobType：
python
new_job = cloudpss_substation_Model.createJob('emtp', name='新的电磁暂态仿真方案')
通用调试建议
检查数据结构：
使用 print(type(obj)) 和 print(obj) 查看对象的类型和内容。
使用 dir(obj) 查看对象的属性和方法。
防御性编程：
在访问字典键或列表索引前，先检查是否存在：
python
if 'key' in my_dict:
    value = my_dict['key']
if len(my_list) > 0:
    item = my_list[0]
查阅文档：
确认 cloudpss_substation_Model 的 API 文档，了解正确的属性和方法用法。
逐步执行：
将复杂操作拆分为小步骤，逐步验证每一步的输出。""",
                "类型错误": "错误主要集中在 TypeError，具体分为以下几种情况：对不兼容对象进行操作：对象不支持字典式赋值：尝试对 Component 类型的对象使用 component['label'] = '...' 这种字典赋值方式，但该对象不支持此操作 。对象不是可调用函数：尝试将一个 ModelRevision 类型的对象当作函数调用，例如 cloudpss_substation_Model.revision()，但它不是一个可调用的函数 。对象不可序列化：尝试将 Model 类型的对象转换为 JSON 格式时失败，因为该对象不具备 JSON 序列化的能力 。索引类型错误：字符串作为索引：在循环遍历列表时，列表中混合了字典和字符串，但代码尝试用 val['name'] 这种方式通过字符串键来访问字符串元素，导致 string indices must be integers 错误 。这在多个场景中反复出现，例如：在 getModelJob 方法中，遍历 self.jobs 列表时，列表中的字符串元素无法通过 'name' 索引访问 。在遍历 all_components 时，component 变量意外地变成了字符串，导致无法使用 'id' 索引 。函数调用参数缺失：在调用 Model.addComponent() 方法时，缺少了一个必需的位置参数 pins 。在调用 requests.Session.request() 方法时，传递了一个它不期望的关键字参数 kwargs，这通常是由于函数调用链中参数处理不当造成的 。调试方法针对上述错误，文件中提供了以下调试方法和建议：检查数据结构：当遇到 string indices must be integers 错误时，建议打印出列表或对象（如 cloudpss_substation_Model.jobs）的完整内容来检查其结构，以确认其中是否混合了不同类型的数据（如字典和字符串） 。通过 print(type(job)) 检查变量的具体数据类型 。修改代码以适应数据结构：使用 isinstance(val, dict) 或 val.get('name') 来判断元素是否为字典，并安全地访问其键值对，从而避免对字符串进行索引操作 。使用列表推导式过滤掉非字典数据 。使用 next() 函数配合条件判断来获取第一个匹配的元素，而不是依赖于可能混合了其他类型数据的列表 。补充或修改函数参数：对于 addComponent() 方法缺少 pins 参数的错误，需要查阅文档或根据经验补充该参数，即使是空列表 [] 。检查对象属性：当遇到对象不支持特定操作或无法序列化时，可以遍历对象的属性（dir(object)），并使用 getattr() 安全地打印每个属性的值，以更好地了解对象的内部结构 。",
                "业务逻辑错误": "业务逻辑错误 → 空数据或资源未找到",
                "完成任务": "技能成功完成预期任务"
            }

            print(f"技能预测器加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"技能预测器加载失败: {str(e)}")
            raise

    def predict_skills(self, state_text: str, skill_texts: List[str]) -> List[Dict]:
        """
        预测多个技能的成功概率和失败原因分布

        Args:
            state_text: 当前状态描述
            skill_texts: 技能描述列表

        Returns:
            包含每个技能预测结果的列表
        """
        try:
            results = []
            state_emb = self.embedding_model.encode(state_text, convert_to_tensor=True).to(self.device)

            for skill_text in skill_texts:
                skill_emb = self.embedding_model.encode(skill_text, convert_to_tensor=True).to(self.device)
                input_tensor = torch.cat((state_emb, skill_emb)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    probs = self.model.predict_proba(input_tensor).squeeze(0)

                skill_result = {
                    "skill_text": skill_text,
                    "probabilities": {},
                    "success_probability": 0.0,
                    "failure_analysis": []
                }

                total_failure_prob = 0.0
                for idx, reason in self.idx_to_reason.items():
                    prob = probs[idx].item()
                    skill_result["probabilities"][reason] = prob

                    if reason == "完成任务":
                        skill_result["success_probability"] = prob
                    else:
                        total_failure_prob += prob

                # 生成失败风险分析
                if total_failure_prob > 0:
                    sorted_failures = sorted(
                        [(r, probs[i].item()) for i, r in self.idx_to_reason.items() if r != "完成任务"],
                        key=lambda x: -x[1]
                    )[:3]

                    for reason, prob in sorted_failures:
                        if prob > 0.01:
                            explanation = self.failure_reason_explanations.get(reason, "未知失败原因")
                            skill_result["failure_analysis"].append({
                                "reason": reason,
                                "probability": round(prob * 100, 1),
                                "explanation": explanation
                            })

                results.append(skill_result)

            return results
        except Exception as e:
            print(f"技能预测过程中出错: {str(e)}")
            return []


class RAGPipeline:
    """RAG管道：检索增强生成"""

    def __init__(self, dataset_id: str, api_key: str, llm_api_key: str,
                 milvus_host: str = "localhost", milvus_port: str = "19530",
                 model_path: str = "models/failure_reason_predictor.pth",
                 embedding_model_path: str = "models/all-MiniLM-L6-v2"):
        """
        初始化RAG管道

        Args:
            dataset_id: 知识库ID
            api_key: 知识库检索API密钥
            llm_api_key: OpenRouter LLM API密钥
            milvus_host: Milvus服务器地址
            milvus_port: Milvus服务器端口
            model_path: 技能预测模型路径
            embedding_model_path: 嵌入模型路径
        """
        self.dataset_id = dataset_id
        self.kb_api_key = api_key
        self.llm_api_key = llm_api_key

        # 初始化技能预测器
        try:
            self.skill_predictor = SkillPredictor(
                model_path=model_path,
                embedding_model_path=embedding_model_path
            )
        except Exception as e:
            print(f"警告: 技能预测器初始化失败 - {str(e)}")
            self.skill_predictor = None

        # 系统提示词模板
        self.system_prompt_template = """
        你是CloudPSS-BOT，一个世界级的Python程序员，一个专业的电力系统仿真专家，可以利用CloudPSS库完成电力系统仿真任务，通过生成代码实现任何目标。
        当你生成代码时，它将在用户的机器上执行。用户已经给了你完全和完整的权限来生成任何必要的代码来完成任务。
        您可以访问知识库，了解如何使用SDK并生成满足任务要求的代码。
        cloudpss_api_url\cloudpss_Token\cloudpss_substation_Model在python代码中初始化。模型为“cloudpss_substation_Model”。
        运行**任何代码**来实现目标，如果一开始你没有成功，就一次又一次地尝试。
        当用户引用文件名时，他们可能会引用当前正在其中执行代码的目录中的现有文件。
        在Markdown中向用户写入消息。
        一般来说，制定计划的来完成复杂任务，并且每次会话只生成一个步骤的代码，用户完成执行后再判断是否需要生成后续代码。 **不要尝试在一个代码块中完成所有事情是至关重要的。**你应该尝试打印相关信息，然后从那里开始，循序渐进，你可能不能够第一次就生成完整代码。
        你能胜任任何工作。
        ‘import cloudpss\nimport os\nCurrentJob = cloudpss.currentJob()\ncloudpss_api_url = 'xxx'\ncloudpss_Token = 'xxx'\ncloudpss_Model = 'xxx'\nos.environ['CLOUDPSS_API_URL'] = cloudpss_api_url\ncloudpss.setToken(cloudpss_Token)\ncloudpss_substation_Model = cloudpss.Model.fetch(cloudpss_Model)")
        ’
        这个代码已经被执行过了。
        请生成满足任务的代码。您提供的任何Python代码都将被执行。注意：如果已经完成了任务不要提供包含例如'''python、 ``` xxx ``` 或者将被视为代码块的代码和内容，仅需回复"已完成任务"。如果未完成任务，请重新生成代码。
        不要生成input相关代码
        用户名：User
        用户操作系统：Windows


        当前问题：{question}

        上下文信息：{context}
        历史消息: {context_messages}
        """

        # 连接到Milvus服务
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port
        )
        self.collection_name = "state_skill_collection"
        self.collection = Collection(self.collection_name)
        self.collection.load()

        # 初始化嵌入模型
        self.embedding_fn = milvus_model.DefaultEmbeddingFunction()

    def retrieve_documents(self, query: str, limit: int = 3) -> List[Dict]:
        """从Milvus向量知识库检索相关文档"""
        try:
            query_vector = self.embedding_fn.encode_queries([query])[0]

            search_params = {
                "metric_type": 'COSINE',
                "params": {"nprobe": 10}
            }
            results = self.collection.search(
                [query_vector.tolist()],
                "vector",
                search_params,
                limit=limit,
                output_fields=["state", "skill", "text"]
            )

            documents = []
            if results:
                for hits in results:
                    for hit in hits:
                        doc_info = {
                            "content": hit.entity.get("text", ""),
                            "score": hit.distance,
                            "source": f"{hit.entity.get('state', '')}-{hit.entity.get('skill', '')}",
                            "state": hit.entity.get("state", ""),
                            "skill": hit.entity.get("skill", "")
                        }
                        documents.append(doc_info)

            return documents

        except Exception as e:
            print(f"检索失败: {str(e)}")
            return []

    def analyze_skills(self, task: str, state: str, documents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """分析所有检索到的技能"""
        if not self.skill_predictor or not documents:
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

    def format_context(self, ranked_docs: List[Dict]) -> str:
        """格式化上下文"""
        if not ranked_docs:
            return "未找到相关技能。"

        context_parts = ["## 可选技能分析\n"]

        for i, doc in enumerate(ranked_docs[:5], 1):
            analysis = doc["analysis"]
            entry = f"""### 技能选项 {i}
**来源文档**: {doc['source']} (相关性分数: {doc['score']:.4f})
**成功概率**: {analysis['success_probability'] * 100:.1f}%
**代码片段**: {doc['content'][:500]}...
"""
            context_parts.append(entry)
        return "\n".join(context_parts)

    def generate_response_stream(self, query: str, context: str, context_messages: str,
                                 temperature: float = 0.3, max_tokens: int = None) -> Generator[Dict, None, None]:
        """调用LLM生成流式回答"""
        # 从环境变量获取max_tokens，如果未设置则使用1000000
        if max_tokens is None:
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1000000"))

        # 从环境变量获取模型名称，支持高输出容量的模型
        model_name = os.getenv("LLM_MODEL_NAME", "google/gemini-2.5-flash")

        system_prompt = self.system_prompt_template.format(
            question=query,
            context=context,
            context_messages=context_messages
        )

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "CloudPSS RAG Interpreter"
        }

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": temperature,

            "stream": True
        }

        try:
            with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as response:
                response.raise_for_status()

                for chunk in response.iter_lines():
                    if chunk:
                        chunk = chunk.decode('utf-8')
                        if chunk.startswith("data:"):
                            chunk_data = chunk[5:].strip()
                            if chunk_data == "[DONE]":
                                continue

                            try:
                                data = json.loads(chunk_data)
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        yield {
                                            "choices": [{
                                                "delta": {"content": choice["delta"]["content"]},
                                                "index": choice.get("index", 0),
                                                "finish_reason": choice.get("finish_reason")
                                            }]
                                        }
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            print(f"LLM调用失败: {str(e)}")
            yield {"error": str(e)}

    def _preprocess_query(self, query: str) -> str:
        """安全处理查询字符串"""
        processed = ''.join(c if ord(c) >= 32 else '-' for c in query)
        processed = processed.replace('"', '')
        processed = re.sub(r'[\\/]+', ' ', processed)
        processed = ' '.join(processed.split())
        return processed.strip()

    def query(self, task: str, state: str, context_messages: str,
              llm_temperature: float = 0.3, stream: bool = False):
        """完整的RAG查询流程"""
        processed_task = self._preprocess_query(task)
        processed_state = self._preprocess_query(state)

        # 检索阶段
        documents = self.retrieve_documents(query=processed_task + processed_state)

        # 技能分析阶段
        ranked_docs, all_analysis = self.analyze_skills(task, state, documents)

        # 上下文格式化
        context = self.format_context(ranked_docs)

        # 生成阶段
        if stream:
            return self.generate_response_stream(
                query=state if state else task,
                context=context,
                context_messages=context_messages,
                temperature=llm_temperature
            )
        else:
            return {
                "query": state if state else task,
                "retrieved_documents": documents,
                "ranked_skills": ranked_docs,
                "context_used": context
            }

    def only_query_LLM(self, user_query: str, context_messages: str,
                       llm_temperature: float = 0.3, stream: bool = False):
        """仅调用LLM（不检索技能库）"""
        if stream:
            return self.generate_response_stream(
                query=user_query,
                context='',
                context_messages=context_messages,
                temperature=llm_temperature
            )
        else:
            return {"query": user_query, "context_used": ''}


if __name__ == "__main__":
    import os

    # 从环境变量获取配置
    LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")

    # 初始化RAG管道
    rag_pipeline = RAGPipeline(
        dataset_id="your-dataset-id",
        api_key="your-kb-api-key",
        llm_api_key=LLM_API_KEY
    )

    # 测试查询
    task = "获取模型中的所有元件"
    print("流式输出示例:")
    print("=" * 50)

    for chunk in rag_pipeline.query(
            task=task,
            state="",
            context_messages='[]',
            stream=True
    ):
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)
