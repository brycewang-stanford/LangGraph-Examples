# LangGraph Examples

**Language**: [English](#english) | [中文](#中文)

---

## English

Welcome to **LangGraph Examples** - a comprehensive collection of multi-agent application examples built with the LangGraph framework.

### 🎯 About This Repository

This repository transforms LangGraph's official notebook-style demos into **structured, production-ready projects**. Each example is converted from Jupyter notebooks into standalone project formats, making it easier for developers to understand, learn, and deploy LangGraph multi-agent systems.

### 🌟 Why This Repository?

**From Notebooks to Production**: While LangGraph's official examples are excellent for learning concepts, they're often presented as Jupyter notebooks. This repository bridges the gap by providing:

- 🏗️ **Structured Project Architecture**: Each example follows a clear, modular structure
- 📦 **Production-Ready Code**: Clean separation of concerns with proper configuration management
- 🚀 **Easy Deployment**: Ready-to-run applications with minimal setup
- 📚 **Comprehensive Documentation**: Bilingual documentation (English/Chinese) with detailed explanations
- 🛠️ **Best Practices**: Implementation of industry-standard coding practices and patterns

### 📁 Project Structure

```
LangGraph-Examples/
├── 📖 README.md                    # This file - project overview
├── 🔧 .gitignore                   # Git ignore patterns
└── 📂 proj1-adaptive-rag/          # Project 1: Adaptive RAG System
    ├── 📋 README.md                # Project-specific documentation
    ├── ⚙️ config.py                # Configuration management
    ├── 🧠 llm_components.py        # LLM agent components
    ├── 🔍 retriever.py             # Vector storage and retrieval
    ├── 🛠️ tools.py                 # External tools integration
    ├── 📊 models.py                # Data models and state definitions
    ├── 🔗 graph_nodes.py           # LangGraph node definitions
    ├── 🌐 graph_edges.py           # LangGraph edge and routing logic
    ├── 🔄 workflow.py              # Main workflow orchestration
    ├── 🚀 main.py                  # Application entry point
    ├── 🎪 demo.py                  # Demo script
    ├── 📄 requirements.txt         # Python dependencies
    ├── 🔐 .env.example             # Environment variables template
    └── 📓 langgraph_adaptive_rag.ipynb  # Original notebook reference
```

### 🚀 Current Examples

#### 1. **Adaptive RAG System** (`proj1-adaptive-rag/`)

**Source**: [LangGraph Official Adaptive RAG Example](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_cohere.ipynb)

A sophisticated multi-agent RAG (Retrieval Augmented Generation) system featuring:

- 🧠 **Intelligent Routing**: Automatically selects between local knowledge base and web search
- 🔄 **Self-Correction**: Document relevance scoring and query rewriting
- 🏗️ **Multi-Agent Architecture**: 7 specialized agents working in coordination
- 🛡️ **Quality Control**: Hallucination detection and answer completeness verification
- 🌐 **Hybrid Information Sources**: Combines vector databases with real-time web search

**Key Features**:
- Dynamic routing based on question analysis
- Iterative quality improvement with feedback loops
- Real-time information retrieval via Tavily API
- Comprehensive state management with GraphState
- Production-ready configuration system

### 🛠️ Quick Start

#### Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key (for web search)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/brycewang-stanford/LangGraph-Examples.git
cd LangGraph-Examples
```

2. **Navigate to a project**
```bash
cd proj1-adaptive-rag
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

5. **Run the example**
```bash
# Run demo examples
python main.py --mode example

# Interactive mode
python main.py --mode interactive

# Single query
python main.py --mode single --question "What are the types of agent memory?"
```

### 🎯 Learning Path

For developers new to LangGraph, we recommend following this learning sequence:

1. **Start with `proj1-adaptive-rag`**: Understand the fundamentals of multi-agent systems
2. **Study the Architecture**: Examine how agents communicate via GraphState
3. **Explore Routing Logic**: Learn conditional edges and decision making
4. **Experiment with Configuration**: Try different models and parameters
5. **Build Your Own**: Use these examples as templates for your applications

### 🤝 Contributing

We welcome contributions! Ways to contribute:

- 🐛 **Bug Reports**: Found an issue? Please open an issue
- 💡 **Feature Requests**: Suggest new examples or improvements
- 📝 **Documentation**: Help improve our documentation
- 🔧 **New Examples**: Convert additional LangGraph notebooks to structured projects

#### Adding New Examples

1. Create a new project directory: `projN-example-name/`
2. Follow the established project structure
3. Include comprehensive README with bilingual documentation
4. Add requirements.txt and .env.example
5. Update this main README with your example

### 📚 Resources

- **LangGraph Official Documentation**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **LangGraph Examples Repository**: [https://github.com/langchain-ai/langgraph/tree/main/examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- **LangChain Documentation**: [https://python.langchain.com/](https://python.langchain.com/)

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- **LangChain Team**: For creating the amazing LangGraph framework
- **Community Contributors**: For feedback and suggestions
- **Original Notebook Authors**: For the foundational examples

---

## 中文

欢迎来到 **LangGraph Examples** - 基于 LangGraph 框架构建的多智能体应用示例集合。

### 🎯 关于本仓库

本仓库将 LangGraph 官方的笔记本样式演示转换为**结构化的、生产就绪的项目**。每个示例都从 Jupyter 笔记本转换为独立的项目格式，使开发者更容易理解、学习和部署 LangGraph 多智能体系统。

### 🌟 为什么选择这个仓库？

**从笔记本到生产环境**：虽然 LangGraph 的官方示例非常适合学习概念，但它们通常以 Jupyter 笔记本的形式呈现。本仓库通过提供以下功能来弥补这一差距：

- 🏗️ **结构化项目架构**：每个示例都遵循清晰的模块化结构
- 📦 **生产就绪代码**：清晰的关注点分离，适当的配置管理
- 🚀 **易于部署**：最小化设置的即用型应用程序
- 📚 **全面文档**：双语文档（英文/中文）及详细说明
- 🛠️ **最佳实践**：实施行业标准编码实践和模式

### 📁 项目结构

```
LangGraph-Examples/
├── 📖 README.md                    # 本文件 - 项目概述
├── 🔧 .gitignore                   # Git 忽略模式
└── 📂 proj1-adaptive-rag/          # 项目1：自适应RAG系统
    ├── 📋 README.md                # 项目特定文档
    ├── ⚙️ config.py                # 配置管理
    ├── 🧠 llm_components.py        # LLM智能体组件
    ├── 🔍 retriever.py             # 向量存储和检索
    ├── 🛠️ tools.py                 # 外部工具集成
    ├── 📊 models.py                # 数据模型和状态定义
    ├── 🔗 graph_nodes.py           # LangGraph节点定义
    ├── 🌐 graph_edges.py           # LangGraph边缘和路由逻辑
    ├── 🔄 workflow.py              # 主要工作流编排
    ├── 🚀 main.py                  # 应用程序入口点
    ├── 🎪 demo.py                  # 演示脚本
    ├── 📄 requirements.txt         # Python依赖项
    ├── 🔐 .env.example             # 环境变量模板
    └── 📓 langgraph_adaptive_rag.ipynb  # 原始笔记本参考
```

### 🚀 当前示例

#### 1. **自适应RAG系统** (`proj1-adaptive-rag/`)

**来源**: [LangGraph官方自适应RAG示例](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_cohere.ipynb)

一个复杂的多智能体RAG（检索增强生成）系统，具有以下特性：

- 🧠 **智能路由**：自动在本地知识库和网络搜索之间选择
- 🔄 **自我纠错**：文档相关性评分和查询重写
- 🏗️ **多智能体架构**：7个专业智能体协调工作
- 🛡️ **质量控制**：幻觉检测和答案完整性验证
- 🌐 **混合信息源**：结合向量数据库与实时网络搜索

**核心特性**：
- 基于问题分析的动态路由
- 带反馈循环的迭代质量改进
- 通过Tavily API进行实时信息检索
- 使用GraphState进行全面状态管理
- 生产就绪的配置系统

### 🛠️ 快速开始

#### 前置要求

- Python 3.8+
- OpenAI API密钥
- Tavily API密钥（用于网络搜索）

#### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/brycewang-stanford/LangGraph-Examples.git
cd LangGraph-Examples
```

2. **进入项目目录**
```bash
cd proj1-adaptive-rag
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境**
```bash
cp .env.example .env
# 在.env文件中编辑您的API密钥
```

5. **运行示例**
```bash
# 运行演示示例
python main.py --mode example

# 交互模式
python main.py --mode interactive

# 单个查询
python main.py --mode single --question "智能体的记忆类型有哪些？"
```

### 🎯 学习路径

对于LangGraph新手开发者，我们建议按照以下学习顺序：

1. **从`proj1-adaptive-rag`开始**：理解多智能体系统的基础
2. **研究架构**：检查智能体如何通过GraphState通信
3. **探索路由逻辑**：学习条件边缘和决策制定
4. **实验配置**：尝试不同的模型和参数
5. **构建自己的应用**：使用这些示例作为您应用程序的模板

### 🤝 贡献

我们欢迎贡献！贡献方式：

- 🐛 **错误报告**：发现问题？请提出issue
- 💡 **功能请求**：建议新示例或改进
- 📝 **文档**：帮助改进我们的文档
- 🔧 **新示例**：将其他LangGraph笔记本转换为结构化项目

#### 添加新示例

1. 创建新项目目录：`projN-example-name/`
2. 遵循既定的项目结构
3. 包含双语文档的全面README
4. 添加requirements.txt和.env.example
5. 在此主README中更新您的示例

### 📚 资源

- **LangGraph官方文档**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **LangGraph示例仓库**: [https://github.com/langchain-ai/langgraph/tree/main/examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- **LangChain文档**: [https://python.langchain.com/](https://python.langchain.com/)

### 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

### 🙏 致谢

- **LangChain团队**：创造了令人惊叹的LangGraph框架
- **社区贡献者**：提供反馈和建议
- **原始笔记本作者**：提供基础示例

---

*LangGraph Examples - 让多智能体开发变得简单易懂 | Making multi-agent development accessible and practical*