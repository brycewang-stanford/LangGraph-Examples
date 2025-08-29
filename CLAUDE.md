# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **LangGraph Examples** - a collection of production-ready multi-agent RAG (Retrieval Augmented Generation) systems built with the LangGraph framework. The repository transforms LangGraph's official notebook-style demos into structured, deployable projects with comprehensive bilingual documentation.

## Project Structure

The repository contains multiple independent projects, each demonstrating different RAG approaches:

- **proj0/**: Original Jupyter notebooks from LangGraph official examples
- **proj1-adaptive-rag/**: Structured Adaptive RAG system with intelligent routing
- **proj2-agentiic-rag/**: Agentic RAG system with autonomous decision-making

Each project follows a consistent modular structure for easy understanding and deployment.

## Development Commands

### proj1-adaptive-rag (Primary Project)

```bash
# Setup
cd proj1-adaptive-rag
pip install -r requirements.txt
cp .env.example .env  # Edit with your API keys

# Run examples
python main.py --mode example           # Demo queries
python main.py --mode interactive       # Interactive chat
python main.py --mode single --question "What are the types of agent memory?"

# Debug mode
python main.py --mode single --question "Your question" --verbose
```

### proj2-agentiic-rag

```bash
# Setup  
cd proj2-agentiic-rag/agentic_rag_project
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key" > .env

# Run
python run.py                           # Interactive mode
python run.py "Your question"           # Single query
python -m src.main "Your question"      # Module execution
```

## Architecture Overview

### LangGraph Multi-Agent Systems

Both projects implement sophisticated multi-agent architectures using LangGraph's StateGraph pattern:

**Key Components:**
- **State Management**: Uses `GraphState` with message accumulation via `add_messages`
- **Node Functions**: Individual processing units (retrieve, grade, generate, etc.)
- **Conditional Edges**: Dynamic routing based on document relevance and quality checks
- **Workflow Compilation**: Graph → executable application via `compile()`

### Adaptive RAG (proj1) Architecture

**7-Agent Coordination System:**
1. **Question Router**: Selects between vectorstore and web search
2. **Document Retriever**: ChromaDB-based vector search  
3. **Web Search**: Tavily API integration
4. **Document Grader**: LLM-based relevance scoring
5. **Answer Generator**: Context-aware response generation
6. **Query Transformer**: Question rewriting for better retrieval
7. **Hallucination Detector**: Quality control and self-correction

**Critical Workflow Pattern:**
```
START → Route Question → [Retrieve | Web Search] → Grade Documents → Generate → Quality Check → [END | Retry | Transform]
```

### Agentic RAG (proj2) Architecture

**Autonomous Decision-Making:**
- **Agent Node**: Makes tool usage decisions using OpenAI models
- **Tool Binding**: LangChain tools integrated via `bind_tools()`
- **State Accumulation**: Message history maintained across workflow steps
- **Answer Extraction**: Only responses from "generate" node become final answers

## Configuration Management

### Environment Variables
Both projects use `.env` files with similar structure:
```bash
# Required
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here  # For web search

# Optional
COHERE_API_KEY=your_key_here  # Alternative embeddings
DEBUG_MODE=false
VERBOSE=true
```

### Model Configuration
- **Default Models**: Both projects use `gpt-5` series (configurable)
- **Temperature**: 0.0 for deterministic responses
- **Embeddings**: OpenAI embeddings (ChromaDB integration)
- **Streaming**: Disabled for compatibility

### Vector Store Settings
- **Backend**: ChromaDB with persistent storage
- **Chunk Size**: 500 tokens (proj1), 100 tokens (proj2)
- **Collection Names**: "rag-chroma" 
- **Default Sources**: Lilian Weng's blog posts on agents, prompt engineering, adversarial attacks

## Key Implementation Patterns

### LangGraph State Management
```python
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    # Uses add_messages for accumulation
```

### Conditional Edge Pattern
```python
workflow.add_conditional_edges(
    "source_node",
    decision_function,  # Returns routing key
    {
        "route_a": "destination_node_a",
        "route_b": "destination_node_b",
    }
)
```

### Tool Integration (proj2)
```python
# Bind tools to LLM for agent decision-making
llm_with_tools = llm.bind_tools(tools)

# Agent makes autonomous tool usage decisions
def agent(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### Lazy Initialization Pattern
Both projects use lazy initialization for dependencies:
```python
def get_retriever(self):
    if self._retriever is None:
        self._retriever = self.vector_store.as_retriever()
    return self._retriever
```

## Data Flow Patterns

### proj1 (Adaptive RAG) Flow
1. Question classification (vectorstore vs web)
2. Source-specific retrieval/search
3. Document relevance grading
4. Context-aware generation
5. Quality validation with retry loops

### proj2 (Agentic RAG) Flow
1. Agent decides on tool necessity
2. Conditional tool execution
3. Document grading and query rewriting
4. Final answer generation
5. **Critical**: Only "generate" node outputs are captured as answers

## Development Guidelines

### Modifying Workflows
- **State Schema**: Changes to `GraphState` affect all nodes
- **Node Signatures**: Must accept and return state dictionaries
- **Edge Conditions**: Routing functions must return exact keys from edge mappings
- **Tool Binding**: Tools must be LangChain-compatible for agent integration

### Adding New Data Sources
1. Update `DEFAULT_URLS` in `config.py`
2. Consider chunk size adjustments for new content types
3. Test document grading with new domain-specific content
4. Verify embedding model performance on new data

### Model Switching
- Update model names in configuration files
- Verify streaming compatibility
- Check token limits for chunk processing
- Test tool binding compatibility (proj2)

### Environment Management
- Each project uses isolated virtual environments
- Dependencies specified in project-level `requirements.txt`
- Configuration validation on startup
- API key validation before workflow execution

## Testing and Validation

### Built-in Examples
Both projects include example queries covering:
- Vector store queries (agent memory, prompt engineering)
- Web search queries (current events, news)
- Error handling scenarios
- Multi-turn conversation patterns

### Debugging Features
- Verbose execution tracing with `--verbose`
- Step-by-step workflow visualization
- State inspection at each node
- Execution path tracking for optimization

## Common Pitfalls

1. **API Keys**: Both OpenAI and Tavily keys required for full functionality
2. **Model Names**: Ensure model availability and permissions
3. **Chunk Sizes**: Too large chunks can exceed context limits
4. **Tool Integration**: Agent tools must return LangChain-compatible responses
5. **State Management**: Messages must be properly accumulated using `add_messages`
6. **Answer Extraction**: Only specific nodes produce final answers (varies by project)

This repository demonstrates production-ready patterns for multi-agent RAG systems with comprehensive error handling, quality control, and modular architecture suitable for real-world deployment.