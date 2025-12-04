# CloudPSS RAG Interpreter

基于检索增强生成(RAG)和技能成功率预测的电力系统仿真代码自动生成系统。

## 项目简介

通过Milvus向量数据库检索相关技能代码，并使用神经网络预测技能执行成功率，从而为CloudPSS电力系统仿真平台自动生成高质量的Python代码。

### 主要特性

- **RAG增强代码生成**: 基于向量检索的上下文增强，提供相关技能代码参考
- **技能成功率预测**: 使用神经网络预测代码执行成功概率，优先选择高成功率技能
- **流式输出**: 支持实时流式响应，提升用户体验
- **自动错误处理**: 智能识别错误并触发技能检索进行修复
- **多轮对话支持**: 保持会话上下文，支持迭代式代码开发

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户请求                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI 服务器                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Open Interpreter                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ 代码执行器   │  │ 会话管理    │  │ 消息处理    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                RAG Pipeline                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Milvus检索  │→│ 技能预测器  │→│ LLM生成器   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Milvus  │   │ OpenRouter│   │ CloudPSS │
        │ 向量数据库│   │   LLM    │   │  平台    │
        └──────────┘   └──────────┘   └──────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- Docker (用于运行Milvus)
- CUDA (可选，用于GPU加速)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/create-my/CloudPSS-RAG-Interpreter.git
cd CloudPSS-RAG-Interpreter
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动Milvus数据库**
```bash
# 使用Docker启动Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.3.3
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥和配置
```

5. **下载模型文件**

将以下模型文件放入 `models/` 目录：
- `failure_reason_predictor.pth` - 技能成功率预测模型
- `all-MiniLM-L6-v2/` - 文本嵌入模型（可从HuggingFace下载）

6. **初始化向量数据库**
```bash
cd src
python milvus_init.py --csv ../data/skills_27.csv
```

7. **启动服务器**
```bash
python server.py
```

### 使用方法

**单次查询**
```bash
python client.py --task "获取模型中的所有元件"
```

**批量测试**
```bash
python client.py --csv ../data/tasks.csv
```

**API调用**
```bash
curl "http://localhost:8000/chat?message=获取模型中的所有元件"
```

## 项目结构

```
CloudPSS-RAG-Interpreter/
├── src/
│   ├── rag_pipeline.py      # RAG管道核心实现
│   ├── server.py            # FastAPI服务器
│   ├── client.py            # 测试客户端
│   └── milvus_init.py       # Milvus初始化工具
├── models/
│   ├── failure_reason_predictor.pth  # 预测模型
│   └── all-MiniLM-L6-v2/             # 嵌入模型
├── data/
│   └── skills.csv           # 技能数据样本
├── docs/
│   └── ...                  # 文档
├── .env.example             # 环境变量模板
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
```

## API文档

### GET /chat

执行代码生成任务

**参数**
- `message` (string): 任务描述

**返回**
- 消息列表，包含生成的代码和执行结果

### GET /history

获取当前会话历史

### GET /health

健康检查

## 技能预测模型

技能预测器使用神经网络分析以下失败原因：

| 失败类型 | 说明 |
|---------|------|
| 属性错误 | 对象缺少属性或方法 |
| 变量或引用错误 | 变量未定义 |
| 键值或索引错误 | 字典/列表访问错误 |
| 类型错误 | 类型不匹配 |
| 业务逻辑错误 | 资源未找到等 |
| 完成任务 | 成功执行 |

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| LLM_API_KEY | OpenRouter API密钥 | - |
| MILVUS_HOST | Milvus服务器地址 | localhost |
| MILVUS_PORT | Milvus端口 | 19530 |
| CLOUDPSS_TOKEN | CloudPSS平台Token | - |
| CLOUDPSS_MODEL | CloudPSS模型ID | - |

## 开发指南

### 添加新技能

1. 准备CSV格式的技能数据：
```csv
state,skill
"用户请求描述","对应的代码实现"
```

2. 导入到Milvus：
```bash
python milvus_init.py --csv your_skills.csv
```

## 注意事项

- 首次运行需要先初始化Milvus数据库（已完成）
- 服务器启动后会自动加载技能预测模型
- 如果看到编码警告或FutureWarning，可以忽略
- 发布前请删除 `.env` 文件

## 常见问题

**Q: 端口被占用怎么办？**
```bash
# Windows
netstat -ano | findstr ":8000"
taskkill /F /PID <进程ID>

# Linux/Mac
lsof -i :8000
kill -9 <进程ID>
```

**Q: Milvus连接失败？**
确保Milvus容器正在运行且端口19530可访问。

**Q: 模型加载失败？**
检查 `models/` 目录下是否有：
- `failure_reason_predictor.pth`
- `all-MiniLM-L6-v2/` 文件夹


## 许可证

MIT License

## 致谢

- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)
- [Milvus](https://milvus.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [CloudPSS](https://cloudpss.net/)

## 联系方式

yuanxuefeng@mail.tsinghua.edu.cn
