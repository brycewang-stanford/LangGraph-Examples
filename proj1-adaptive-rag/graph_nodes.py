"""
图节点模块

这个模块定义了 LangGraph 工作流中的所有节点函数。
每个节点代表工作流中的一个处理步骤。

LangGraph 核心概念：
- 节点（Node）：执行特定任务的函数
- 节点函数接收状态，执行操作，返回更新后的状态
- 节点之间通过状态传递信息
"""

from typing import Dict, Any
from langchain_core.documents import Document

from models import GraphState
from llm_components import LLMComponents
from tools import ToolManager


class GraphNodes:
    """
    图节点管理类
    
    封装所有节点函数，提供工作流的核心处理逻辑。
    这是 LangGraph 应用的核心部分，定义了数据如何在图中流动和转换。
    """
    
    def __init__(
        self,
        retriever,
        llm_components: LLMComponents,
        tool_manager: ToolManager
    ):
        """
        初始化图节点
        
        参数:
            retriever: 文档检索器
            llm_components: LLM 组件管理器
            tool_manager: 工具管理器
        """
        self.retriever = retriever
        self.llm_components = llm_components
        self.tool_manager = tool_manager
        
        # 创建所有需要的 LLM 链
        self.rag_chain = llm_components.create_rag_chain()
        self.retrieval_grader = llm_components.create_retrieval_grader()
        self.question_rewriter = llm_components.create_question_rewriter()
    
    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """
        检索节点
        
        从向量存储中检索相关文档。
        这是 RAG 流程的第一步，获取可能包含答案的文档。
        
        LangGraph 知识点：
        - 节点函数接收当前状态
        - 返回的字典会更新图状态
        - 只返回需要更新的字段
        
        参数:
            state: 当前图状态
            
        返回:
            包含更新字段的字典
        """
        print("---检索文档---")
        question = state["question"]
        
        # 使用检索器获取相关文档
        documents = self.retriever.invoke(question)
        
        # 返回更新的状态字段
        # LangGraph 会自动合并这些更新到全局状态
        return {"documents": documents, "question": question}
    
    def generate(self, state: GraphState) -> Dict[str, Any]:
        """
        生成节点
        
        基于检索到的文档生成答案。
        这是 RAG 的核心步骤，将检索结果转化为连贯的答案。
        
        参数:
            state: 当前图状态
            
        返回:
            包含生成答案的更新状态
        """
        print("---生成答案---")
        question = state["question"]
        documents = state["documents"]
        
        # 调用 RAG 链生成答案
        # 将文档和问题作为上下文传递给 LLM
        generation = self.rag_chain.invoke({
            "context": documents,
            "question": question
        })
        
        # 更新状态，添加生成的答案
        return {
            "documents": documents,
            "question": question,
            "generation": generation
        }
    
    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        文档评分节点
        
        评估检索到的文档是否与问题相关。
        这是自纠正 RAG 的关键步骤，确保只使用相关文档。
        
        LangGraph 知识点：
        - 节点可以过滤和修改状态中的数据
        - 这种评分机制实现了质量控制
        
        参数:
            state: 当前图状态
            
        返回:
            包含过滤后文档的更新状态
        """
        print("---检查文档相关性---")
        question = state["question"]
        documents = state["documents"]
        
        # 评分每个文档
        filtered_docs = []
        for d in documents:
            # 调用相关性评分器
            score = self.retrieval_grader.invoke({
                "question": question,
                "document": d.page_content
            })
            
            grade = score.binary_score
            if grade == "yes":
                print("---评分：文档相关---")
                filtered_docs.append(d)
            else:
                print("---评分：文档不相关---")
                continue
        
        # 返回过滤后的文档
        return {"documents": filtered_docs, "question": question}
    
    def transform_query(self, state: GraphState) -> Dict[str, Any]:
        """
        查询转换节点
        
        重写问题以改善检索效果。
        当初始检索失败时，通过重写问题来获得更好的结果。
        
        LangGraph 知识点：
        - 节点可以修改状态中的任何字段
        - 这种自适应机制提高了系统的鲁棒性
        
        参数:
            state: 当前图状态
            
        返回:
            包含重写问题的更新状态
        """
        print("---转换查询---")
        question = state["question"]
        documents = state["documents"]
        
        # 使用 LLM 重写问题
        better_question = self.question_rewriter.invoke({"question": question})
        
        print(f"原始问题: {question}")
        print(f"重写后的问题: {better_question}")
        
        # 更新问题字段
        return {"documents": documents, "question": better_question}
    
    def web_search(self, state: GraphState) -> Dict[str, Any]:
        """
        网络搜索节点
        
        执行网络搜索以获取最新信息。
        当问题需要实时信息时使用此节点。
        
        LangGraph 知识点：
        - 节点可以调用外部工具和 API
        - 这展示了 LangGraph 与外部系统的集成能力
        
        参数:
            state: 当前图状态
            
        返回:
            包含网络搜索结果的更新状态
        """
        print("---网络搜索---")
        question = state["question"]
        
        # 执行网络搜索
        web_results = self.tool_manager.search_web(question)
        
        # 将网络搜索结果作为文档返回
        # 统一格式，便于后续处理
        return {"documents": web_results, "question": question}
    
    def format_final_output(self, state: GraphState) -> str:
        """
        格式化最终输出
        
        将生成的答案格式化为用户友好的格式。
        
        参数:
            state: 最终图状态
            
        返回:
            格式化的答案字符串
        """
        generation = state.get("generation", "未能生成答案")
        question = state.get("question", "")
        
        # 构建格式化的输出
        output = f"""
问题：{question}

答案：
{generation}
"""
        return output
    
    def debug_state(self, state: GraphState):
        """
        调试辅助函数
        
        打印当前状态信息，用于调试和监控。
        
        参数:
            state: 当前图状态
        """
        print("\n=== 当前状态 ===")
        print(f"问题: {state.get('question', 'N/A')}")
        print(f"文档数量: {len(state.get('documents', []))}")
        print(f"是否有生成: {'是' if state.get('generation') else '否'}")
        print("================\n")