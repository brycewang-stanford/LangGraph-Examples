# Agentic RAG System

一个基于 LangGraph 的智能检索增强生成（Agentic RAG）系统，具有自主决策能力的 AI 代理。

## 🚀 项目概述

Agentic RAG 是传统 RAG（检索增强生成）的智能化升级版本。与简单的"检索-生成"流程不同，本系统能够：

- **智能决策**：自主判断是否需要检索文档
- **质量评估**：评估检索文档的相关性
- **查询优化**：自动重写查询以获得更好的结果
- **多轮推理**：支持复杂的多步推理过程

## 🏗️ 系统架构

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   用户查询   │ ──→│   Agent 决策  │ ──→│   检索文档   │
└─────────────┘    └──────────────┘    └─────────────┘
                           │                    │
                           ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   直接回答   │ ←──│   文档评估    │ ←──│   查询重写   │
└─────────────┘    └──────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │   生成答案   │
                   └─────────────┘
```

## 📁 项目结构

```
agentic_rag_project/
├── src/
│   ├── __init__.py          # 包初始化
│   ├── config.py            # 配置管理
│   ├── vector_store.py      # 向量存储和检索
│   ├── tools.py             # 工具定义
│   ├── nodes.py             # 工作流节点
│   ├── workflow.py          # 工作流构建
│   └── main.py              # 主程序入口
├── .env.example             # 环境变量模板
├── requirements.txt         # 依赖清单
└── README.md               # 项目说明
```

## 🛠️ 安装部署

### 1. 环境要求

- Python 3.8+
- OpenAI API 密钥（支持 gpt-5 模型）

### 2. 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd agentic_rag_project

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量，设置 OpenAI API 密钥
# OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 快速开始

### 命令行模式

```bash
# 单次查询
python -m src.main "What does Lilian Weng say about agent memory?"

# 或者直接运行
cd src && python main.py "您的问题"
```

### 交互式模式

```bash
cd src && python main.py
```

然后按提示输入问题，输入 `quit` 或 `exit` 退出。

### 编程方式使用

```python
from src import AgenticRAGSystem

# 创建系统实例
rag_system = AgenticRAGSystem()

# 初始化系统
rag_system.initialize()

# 单次查询
answer = rag_system.query("What are the types of agent memory?")
print(answer)

# 批量查询
questions = [
    "What is prompt engineering?",
    "How do adversarial attacks work on LLMs?",
    "What are the components of an agent system?"
]
results = rag_system.batch_query(questions)
for question, answer in results.items():
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

## ⚙️ 核心组件

### 1. 配置管理 (config.py)
- 集中管理所有系统配置
- 统一使用 gpt-5 模型
- 支持环境变量配置

### 2. 向量存储 (vector_store.py)
- 文档加载和预处理
- 向量化存储（使用 ChromaDB）
- 语义相似度检索

### 3. 工作流节点 (nodes.py)
- **Agent 节点**：智能决策是否检索
- **文档评估**：判断检索结果相关性
- **查询重写**：优化搜索查询
- **答案生成**：基于相关文档生成回答

### 4. 工作流编排 (workflow.py)
- 使用 LangGraph 构建状态图
- 条件路由和智能分支
- 支持复杂的多步推理

## 🎯 使用示例

### 示例 1：基础问答
```
用户输入：What does Lilian Weng say about the types of agent memory?

系统处理：
1. Agent 决定需要检索
2. 检索相关文档
3. 评估文档相关性：相关
4. 生成答案

输出：Lilian Weng discusses short-term and long-term memory in agent systems. Short-term memory is used for in-context learning, while long-term memory allows agents to retain and recall information over extended periods.
```

### 示例 2：查询重写
```
用户输入：agent stuff

系统处理：
1. Agent 决定需要检索
2. 检索文档：找到不够相关的内容
3. 评估文档相关性：不相关
4. 重写查询：agent components and architecture
5. 重新检索和生成答案
```

## 🔧 自定义配置

### 修改数据源
在 `config.py` 中修改 `BLOG_URLS` 列表：

```python
BLOG_URLS = [
    "https://your-blog-1.com",
    "https://your-blog-2.com",
    # 添加更多数据源
]
```

### 调整模型参数
```python
# 在 config.py 中修改
DEFAULT_TEMPERATURE = 0.1  # 调整创造性
CHUNK_SIZE = 200          # 调整文档块大小
CHUNK_OVERLAP = 100       # 调整重叠大小
```

### 自定义工具
在 `tools.py` 中添加新的工具：

```python
def create_custom_tool():
    # 实现自定义工具逻辑
    pass
```

## 📊 性能优化

### 1. 向量存储优化
- 使用合适的 chunk_size 和 overlap
- 考虑使用更高效的向量数据库（如 Pinecone、Weaviate）

### 2. 模型选择
- 根据任务复杂度选择合适的模型
- 平衡成本和性能

### 3. 缓存机制
- 实现查询结果缓存
- 避免重复计算

## 🚨 注意事项

1. **API 密钥安全**：确保 `.env` 文件不被提交到版本控制
2. **模型调用费用**：gpt-5 模型调用可能产生费用，请注意使用量
3. **网络连接**：系统需要网络连接来访问 OpenAI API 和加载文档
4. **文档权限**：确保有权限访问配置的文档 URL

## 🔄 扩展开发

### 添加新节点
1. 在 `nodes.py` 中定义新的节点函数
2. 在 `workflow.py` 中添加节点到图中
3. 配置适当的边和条件

### 集成新的数据源
1. 在 `vector_store.py` 中添加新的加载器
2. 更新配置以支持新的数据源类型

### 自定义评估逻辑
1. 修改 `nodes.py` 中的 `grade_documents` 函数
2. 实现自定义的相关性评估算法

## 📝 许可证

MIT License

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📞 支持与反馈

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 邮件联系
- 项目讨论区

---

**享受智能问答的乐趣！** 🎉