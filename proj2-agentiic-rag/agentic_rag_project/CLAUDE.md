# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Agentic RAG (Retrieval Augmented Generation) system built with LangGraph that demonstrates intelligent document retrieval and generation. Unlike traditional RAG systems, this implementation uses an AI agent that makes autonomous decisions about when to retrieve documents, evaluates document relevance, and can rewrite queries for better results.

## Running the System

### Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Run Commands
```bash
# Interactive mode
python run.py

# Single query mode
python run.py "What does Lilian Weng say about agent memory?"

# Alternative (module execution)
python -m src.main "Your question here"
```

### Development Commands
```bash
# Run with verbose output (shows workflow execution)
cd src && python main.py "Your question"

# Test batch queries programmatically
python -c "
from src.main import AgenticRAGSystem
system = AgenticRAGSystem()
system.initialize()
system.batch_query(['question1', 'question2'])
"
```

## Architecture Overview

### Core Components
- **Agent Node** (`nodes.py:agent`): Makes decisions about tool usage using OpenAI models
- **Document Grader** (`nodes.py:grade_documents`): Evaluates retrieved document relevance
- **Query Rewriter** (`nodes.py:rewrite`): Improves queries when documents are irrelevant
- **Answer Generator** (`nodes.py:generate`): Creates final responses using RAG prompts
- **Vector Store** (`vector_store.py`): ChromaDB-based document indexing and retrieval
- **Workflow Engine** (`workflow.py`): LangGraph state machine orchestrating the flow

### Critical Workflow Logic
The system follows this exact decision flow (DO NOT MODIFY):
1. User query → Agent decides whether to use retrieval tools
2. If tools needed: Retrieve documents → Grade relevance → Generate/Rewrite
3. If tools not needed: Direct END (results in "未能生成答案")
4. **Answer Extraction**: Only responses from the "generate" node are captured as final answers

### State Management
- Uses LangGraph's `StateGraph` with `AgentState` containing message history
- Messages are accumulated using `add_messages` annotation
- Conditional edges determine workflow paths based on agent decisions and document grading

## Configuration

### Model Configuration (`config.py`)
- **Current Models**: All components use `gpt-5-mini` (configurable)
- **Temperature**: 0.0 (deterministic responses)
- **Streaming**: Disabled for compatibility
- **Environment**: Loads from `.env` using python-dotenv

### Document Sources
Default URLs target Lilian Weng's blog posts about:
- LLM Agents
- Prompt Engineering  
- Adversarial Attacks on LLMs

Modify `BLOG_URLS` in `config.py` to change data sources.

### Vector Store Settings
- **Chunk Size**: 100 tokens (small chunks for precise retrieval)
- **Chunk Overlap**: 50 tokens
- **Collection**: "rag-chroma"
- **Embeddings**: OpenAI embeddings

## Key Implementation Details

### Lazy Initialization Pattern
Tools and vector stores use lazy initialization to handle dependency order:
```python
def get_tools(self):
    if self.tools is None:
        self.tools = tools_manager.get_tools()
    return self.tools
```

### Module vs Script Execution
- `run.py` handles path setup for standalone execution
- Direct module execution: `python -m src.main`
- Script execution: `python run.py`

### Answer Extraction Logic (CRITICAL)
The system ONLY captures answers from the "generate" node:
```python
if key == "generate" and "messages" in value:
    final_answer = value["messages"][0]
```
This means unrelated questions result in "未能生成答案" when the agent doesn't call tools.

### Error Handling
- Configuration validation on startup
- Graceful handling of initialization failures
- Proper exception propagation in query processing

## File Structure Logic

### Core Modules
- `config.py`: Centralized configuration with environment variable loading
- `vector_store.py`: Document loading, chunking, and ChromaDB integration  
- `tools.py`: LangChain tool creation and management
- `nodes.py`: Individual workflow node implementations
- `workflow.py`: LangGraph state machine construction
- `main.py`: System orchestration and user interface

### Entry Points
- `run.py`: Standalone script execution (recommended)
- `src/main.py`: Module entry point with CLI argument parsing

### Dependencies
- **LangChain Ecosystem**: Core framework, OpenAI integration, community tools
- **LangGraph**: Workflow orchestration and state management
- **ChromaDB**: Vector storage backend
- **Python-dotenv**: Environment configuration

## Development Guidelines

### Modifying Workflow Logic
The workflow logic in `workflow.py` implements a specific state machine. Changes require understanding:
- Conditional edge logic (`tools_condition`, `grade_documents`)
- Node interconnections and state passing
- Tool binding and execution patterns

### Adding New Nodes
1. Implement node function in `nodes.py` following the signature pattern
2. Add node to workflow in `workflow.py` 
3. Define appropriate edges and conditions
4. Update state handling if needed

### Extending Data Sources
1. Modify `BLOG_URLS` in `config.py`
2. Consider custom document loaders in `vector_store.py`
3. Adjust chunking parameters for different content types

### Model Configuration Changes
Update model names in `config.py` but ensure:
- Streaming compatibility 
- Token limits for chunk sizes
- API key permissions for selected models