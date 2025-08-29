# LangGraph Agentic RAG 案例详解

## 项目概述

本项目展示了如何使用 LangGraph 构建一个智能化的 RAG（检索增强生成）系统。与传统 RAG 不同，这个系统具有智能决策能力，能够**自主判断是否需要检索**、评估文档相关性，并在必要时重写查询。

## 核心概念

### 什么是 Agentic RAG？

Agentic RAG 是将智能代理（Agent）与检索增强生成相结合的架构。系统不是简单地检索和生成，而是：
- 智能判断是否需要检索
- 评估检索结果的相关性
- 必要时重写查询以获得更好的结果
- 基于相关文档生成高质量答案

## 系统架构

### 1. 环境设置

```python
# 安装必要的依赖包
%pip install -U langchain-community tiktoken langchain-openai langchainhub chromadb langchain langgraph langchain-text-splitters

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### 2. 文档索引构建

系统首先加载并索引了三篇 Lilian Weng 的博客文章：

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",      # LLM 代理
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",  # 提示工程
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",     # 对抗攻击
]
```

**处理流程：**
1. **文档加载**：使用 `WebBaseLoader` 从网页抓取内容
2. **文本分割**：使用 `RecursiveCharacterTextSplitter` 将长文本分割成小块
   - chunk_size=100：每块100个token
   - chunk_overlap=50：块之间50个token的重叠
3. **向量化存储**：使用 Chroma 向量数据库存储文档嵌入
4. **创建检索器**：将向量存储转换为检索器工具

### 3. 检索工具创建

```python
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "搜索并返回关于 LLM 代理、提示工程和对抗攻击的信息"
)
```

这个工具让 Agent 能够根据查询检索相关文档。

## 核心组件详解

### 1. Agent State（代理状态） $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**状态管理：**
- 使用消息列表作为状态载体
- `add_messages` 函数确保消息被追加而非替换
- 每个节点都会更新这个共享状态

### 2. 节点功能

#### Agent 节点
```python
def agent(state):
    """决策节点：判断是否需要调用检索工具"""
    # 使用 GPT-4-turbo 模型
    # 绑定检索工具
    # 根据当前对话决定下一步动作
```

**功能：** 
- 接收用户问题
- 决定是否需要检索文档
- 可以直接回答或调用检索工具

#### Retrieve 节点
```python
retrieve = ToolNode([retriever_tool])
```

**功能：**
- 执行实际的文档检索
- 返回相关文档片段

#### Grade Documents 节点（条件边）
```python
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """评估检索文档的相关性"""
    # 使用结构化输出评估相关性
    # 返回 "yes" 或 "no" 的二元评分
```

**评估流程：**
1. 提取用户问题和检索到的文档
2. 使用 GPT-4o 评估文档相关性
3. 相关则进入生成阶段，不相关则重写查询

#### Rewrite 节点
```python
def rewrite(state):
    """重写查询以获得更好的检索结果"""
    # 分析原始问题的语义意图
    # 生成改进的查询
```

**重写策略：**
- 理解用户的潜在意图
- 优化查询表述
- 提高检索精度

#### Generate 节点
```python
def generate(state):
    """基于相关文档生成最终答案"""
    # 使用 RAG 提示模板
    # 结合上下文生成答案
```

**生成策略：**
- 使用标准 RAG 提示模板
- 限制答案长度（最多3句话）
- 保持答案简洁准确

### 3. 工作流程图

```python
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# 添加边和条件边
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, ...)
workflow.add_conditional_edges("retrieve", grade_documents)
```

**执行流程：** $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
## What does Lilian Weng say about the types of agent memory?
## agent stuff
```
开始 → Agent决策 → 
  ├─ 需要检索 → 检索文档 → 评估相关性 →
  │   ├─ 相关 → 生成答案 → 结束
  │   └─ 不相关 → 重写查询 → Agent决策（循环）
  └─ 不需要检索 → 结束
```
```
  1. Agent 决定是否调用工具：如果不调用工具，直接到 END
  2. 如果调用工具：retrieve → grade_documents → (generate 或 rewrite)
  3. 关键点：对于不相关问题，Agent 不调用工具，直接结束，不会生成任何答案
```

## 实际运行示例

### 输入查询
```python
inputs = {
    "messages": [
        ("user", "What does Lilian Weng say about the types of agent memory?"),
    ]
}
```

### 执行过程追踪

1. **CALL AGENT**
   - Agent 决定需要检索
   - 调用 `retrieve_blog_posts` 工具
   - 查询参数："types of agent memory"

2. **CHECK RELEVANCE**
   - 评估检索到的文档
   - 判定：文档相关（DOCS RELEVANT）

3. **RETRIEVE**
   - 返回包含记忆类型的文档片段
   - 包含短期记忆和长期记忆的描述

4. **GENERATE**
   - 基于相关文档生成答案
   - 输出：Lilian Weng 讨论了代理系统中的短期和长期记忆

### 最终答案
> "Lilian Weng discusses short-term and long-term memory in agent systems. Short-term memory is used for in-context learning, while long-term memory allows agents to retain and recall information over extended periods."

## 关键优势

### 1. 智能决策
- 不是每次都检索，而是智能判断
- 减少不必要的检索开销

### 2. 质量保证
- 文档相关性评估
- 查询重写机制
- 确保答案基于相关内容

### 3. 灵活性
- 可扩展的节点结构
- 条件路由
- 支持复杂的决策逻辑

### 4. 可观察性
- 清晰的执行日志
- 每步决策都有记录
- 便于调试和优化

## 技术要点

### 向量检索
- 使用 OpenAI Embeddings 进行文本嵌入
- Chroma 作为向量数据库
- 支持语义相似度搜索

### 模型选择
- **Agent**: GPT-4-turbo（需要强大的推理能力）
- **评估**: GPT-4o（需要准确的判断）
- **重写**: GPT-4-0125-preview（需要理解语义）
- **生成**: GPT-3.5-turbo（成本效益平衡）

### 提示工程
- 使用 LangChain Hub 的标准 RAG 提示
- 结构化输出确保评估的一致性
- 清晰的任务指令

## 应用场景

1. **智能问答系统**
   - 企业知识库查询
   - 技术文档助手

2. **研究辅助**
   - 学术论文检索
   - 文献综述生成

3. **客户支持**
   - 智能FAQ系统
   - 产品文档查询

## 扩展可能性

1. **多轮对话支持**
   - 保持对话上下文
   - 支持追问和澄清

2. **多源检索**
   - 集成多个数据源
   - 跨库检索协调

3. **缓存优化**
   - 查询结果缓存
   - 重复问题快速响应

4. **反馈循环**
   - 用户反馈收集
   - 模型持续优化

## 总结

这个 Agentic RAG 系统展示了如何将传统 RAG 架构升级为具有智能决策能力的系统。通过 LangGraph 的状态管理和条件路由，系统能够：

- 智能判断检索需求
- 评估和优化检索质量
- 生成准确、相关的答案

这种架构特别适合需要高质量、可解释答案的应用场景，是构建下一代智能问答系统的重要参考。