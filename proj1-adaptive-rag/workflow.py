"""
工作流模块

这个模块构建和编译 LangGraph 工作流。
它将所有节点和边缘组合成一个完整的图结构。

LangGraph 核心概念：
- StateGraph：定义工作流的图结构
- 节点通过 add_node 添加
- 边缘通过 add_edge 和 add_conditional_edges 添加
- compile() 创建可执行的应用
"""

from typing import Dict, Any, Optional
from langgraph.graph import END, StateGraph, START

from models import GraphState
from graph_nodes import GraphNodes
from graph_edges import GraphEdges
from llm_components import LLMComponents
from retriever import VectorStoreManager
from tools import ToolManager


class AdaptiveRAGWorkflow:
    """
    自适应 RAG 工作流
    
    构建和管理完整的 LangGraph 工作流。
    这是整个系统的核心，定义了数据处理的完整流程。
    
    LangGraph 架构说明：
    1. 问题路由：决定使用向量存储还是网络搜索
    2. 检索/搜索：获取相关信息
    3. 文档评分：过滤不相关文档
    4. 答案生成：基于文档生成答案
    5. 质量控制：检查幻觉和答案质量
    6. 自纠正：必要时重写问题或重新生成
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        llm_components: Optional[LLMComponents] = None,
        tool_manager: Optional[ToolManager] = None
    ):
        """
        初始化工作流
        
        参数:
            vector_store_manager: 向量存储管理器
            llm_components: LLM 组件管理器
            tool_manager: 工具管理器
        """
        # 初始化组件，如果未提供则创建默认实例
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.llm_components = llm_components or LLMComponents()
        self.tool_manager = tool_manager or ToolManager()
        
        # 获取检索器
        self.retriever = self.vector_store_manager.get_retriever()
        
        # 创建节点和边缘管理器
        self.nodes = GraphNodes(
            retriever=self.retriever,
            llm_components=self.llm_components,
            tool_manager=self.tool_manager
        )
        
        self.edges = GraphEdges(llm_components=self.llm_components)
        
        # 编译后的应用
        self.app = None
    
    def build_graph(self) -> StateGraph:
        """
        构建 LangGraph 图结构
        
        这是 LangGraph 的核心方法，定义了整个工作流的结构。
        
        LangGraph 知识点：
        - StateGraph 是有状态的图结构
        - 节点是处理单元
        - 边缘定义节点之间的连接
        - 条件边缘实现动态路由
        
        返回:
            构建好的 StateGraph 实例
        """
        print("正在构建 LangGraph 工作流...")
        
        # 创建状态图
        # GraphState 定义了在图中传递的数据结构
        workflow = StateGraph(GraphState)
        
        # ===== 添加节点 =====
        # 每个节点执行特定的任务
        
        # 网络搜索节点
        workflow.add_node("web_search", self.nodes.web_search)
        
        # 文档检索节点
        workflow.add_node("retrieve", self.nodes.retrieve)
        
        # 文档评分节点
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        
        # 答案生成节点
        workflow.add_node("generate", self.nodes.generate)
        
        # 查询转换节点
        workflow.add_node("transform_query", self.nodes.transform_query)
        
        # ===== 构建图结构 =====
        
        # 1. 入口点：条件路由
        # START 是特殊节点，表示工作流的开始
        workflow.add_conditional_edges(
            START,  # 从开始节点
            self.edges.route_question,  # 路由函数
            {
                "web_search": "web_search",  # 如果返回 "web_search"，去网络搜索节点
                "vectorstore": "retrieve",   # 如果返回 "vectorstore"，去检索节点
            },
        )
        
        # 2. 网络搜索后直接生成答案
        workflow.add_edge("web_search", "generate")
        
        # 3. 检索后进行文档评分
        workflow.add_edge("retrieve", "grade_documents")
        
        # 4. 文档评分后的条件路由
        workflow.add_conditional_edges(
            "grade_documents",
            self.edges.decide_to_generate,
            {
                "transform_query": "transform_query",  # 文档不相关，转换查询
                "generate": "generate",                 # 文档相关，生成答案
            },
        )
        
        # 5. 查询转换后重新检索
        workflow.add_edge("transform_query", "retrieve")
        
        # 6. 生成答案后的质量控制
        workflow.add_conditional_edges(
            "generate",
            self.edges.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",      # 有幻觉，重新生成
                "useful": END,                     # 答案好，结束
                "not useful": "transform_query",   # 答案不好，转换查询
            },
        )
        
        print("工作流构建完成！")
        return workflow
    
    def compile(self) -> Any:
        """
        编译工作流
        
        将图结构编译成可执行的应用。
        
        LangGraph 知识点：
        - compile() 将图转换为可执行的应用
        - 编译后的应用可以通过 stream() 或 invoke() 运行
        
        返回:
            编译后的应用
        """
        if self.app is None:
            workflow = self.build_graph()
            self.app = workflow.compile()
            print("工作流编译成功！")
        return self.app
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        运行工作流
        
        执行完整的 RAG 流程。
        
        参数:
            question: 用户问题
            
        返回:
            包含答案的结果字典
        """
        # 确保应用已编译
        if self.app is None:
            self.compile()
        
        # 准备输入
        inputs = {"question": question}
        
        print(f"\n{'='*50}")
        print(f"处理问题: {question}")
        print(f"{'='*50}\n")
        
        # 运行工作流并收集结果
        final_state = None
        for output in self.app.stream(inputs):
            for key, value in output.items():
                print(f"\n节点 '{key}' 执行完成")
                final_state = value
        
        # 返回最终状态
        return final_state
    
    def run_with_details(self, question: str) -> Dict[str, Any]:
        """
        运行工作流并返回详细信息
        
        提供更详细的执行过程信息。
        
        参数:
            question: 用户问题
            
        返回:
            包含详细执行信息的结果
        """
        # 确保应用已编译
        if self.app is None:
            self.compile()
        
        # 准备输入
        inputs = {"question": question}
        
        # 收集执行轨迹
        execution_trace = []
        final_state = None
        
        print(f"\n{'='*50}")
        print(f"详细执行追踪")
        print(f"问题: {question}")
        print(f"{'='*50}\n")
        
        for output in self.app.stream(inputs):
            for node_name, node_state in output.items():
                # 记录执行轨迹
                trace_entry = {
                    "node": node_name,
                    "question": node_state.get("question"),
                    "has_documents": bool(node_state.get("documents")),
                    "has_generation": bool(node_state.get("generation"))
                }
                execution_trace.append(trace_entry)
                
                # 打印节点信息
                print(f"\n--- 节点: {node_name} ---")
                if node_state.get("generation"):
                    print(f"生成了答案")
                if node_state.get("documents"):
                    print(f"文档数量: {len(node_state.get('documents', []))}")
                
                final_state = node_state
        
        # 构建详细结果
        result = {
            "question": question,
            "answer": final_state.get("generation", "未能生成答案"),
            "execution_trace": execution_trace,
            "final_state": final_state
        }
        
        print(f"\n{'='*50}")
        print("执行完成！")
        print(f"最终答案: {result['answer'][:100]}...")
        print(f"{'='*50}\n")
        
        return result
    
    def visualize_graph(self) -> str:
        """
        生成图结构的文本表示
        
        用于理解工作流结构。
        
        返回:
            图结构的文本描述
        """
        description = """
        Adaptive RAG 工作流结构:
        
        START
          |
          v
        [路由问题]
          /     \\
         /       \\
        v         v
    [网络搜索]  [检索文档]
        |         |
        |         v
        |    [评分文档]
        |       /    \\
        |      /      \\
        |     v        v
        |  [转换查询]  |
        |     |        |
        |     v        |
        |  [检索文档]  |
        |              |
        v              v
        [====生成答案====]
                |
                v
          [质量控制]
           /   |   \\
          /    |    \\
         v     v     v
      [结束] [重试] [转换]
        
        节点说明:
        - 路由问题: 决定使用网络搜索还是向量存储
        - 网络搜索: 搜索最新的网络信息
        - 检索文档: 从向量存储检索相关文档
        - 评分文档: 评估文档相关性
        - 转换查询: 重写问题以改善检索
        - 生成答案: 基于文档生成答案
        - 质量控制: 检查幻觉和答案质量
        """
        return description