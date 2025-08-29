"""工作流构建模块 - 定义和构建 LangGraph 工作流"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from .nodes import agent_nodes
from .tools import tools_manager


class AgentState(TypedDict):
    """代理状态定义"""
    # add_messages 函数定义如何处理更新
    # 默认是替换，add_messages 表示 "追加"
    messages: Annotated[Sequence[BaseMessage], add_messages]


class WorkflowBuilder:
    """工作流构建器类"""
    
    def __init__(self):
        """初始化工作流构建器"""
        self.state_graph = None
        self.compiled_graph = None
        self.tools = None  # 延迟初始化
        self.nodes = agent_nodes
    
    def get_tools(self):
        """获取工具列表（延迟初始化）"""
        if self.tools is None:
            self.tools = tools_manager.get_tools()
        return self.tools
    
    def build_workflow(self) -> StateGraph:
        """
        构建工作流图
        
        Returns:
            构建的状态图
        """
        print("=== 构建工作流 ===")
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        print("添加节点:")
        workflow.add_node("agent", self.nodes.agent)
        print("  - agent: 代理决策节点")
        
        # 创建检索工具节点
        retrieve = ToolNode(self.get_tools())
        workflow.add_node("retrieve", retrieve)
        print("  - retrieve: 文档检索节点")
        
        workflow.add_node("rewrite", self.nodes.rewrite)
        print("  - rewrite: 查询重写节点")
        
        workflow.add_node("generate", self.nodes.generate)
        print("  - generate: 答案生成节点")
        
        # 添加边
        print("添加边和条件边:")
        
        # 从开始到代理节点
        workflow.add_edge(START, "agent")
        print("  - START → agent")
        
        # 代理节点的条件边：决定是否检索
        workflow.add_conditional_edges(
            "agent",
            # 评估代理决策
            tools_condition,
            {
                # 将条件输出映射到图中的节点
                "tools": "retrieve",  # 如果需要工具，则检索
                END: END,            # 如果不需要工具，则结束
            },
        )
        print("  - agent → [条件] → retrieve/END")
        
        # 检索节点的条件边：评估文档相关性
        workflow.add_conditional_edges(
            "retrieve",
            # 评估文档相关性
            self.nodes.grade_documents,
        )
        print("  - retrieve → [文档评估] → generate/rewrite")
        
        # 生成节点到结束
        workflow.add_edge("generate", END)
        print("  - generate → END")
        
        # 重写节点回到代理节点
        workflow.add_edge("rewrite", "agent")
        print("  - rewrite → agent")
        
        self.state_graph = workflow
        print("=== 工作流构建完成 ===\n")
        
        return workflow
    
    def compile_workflow(self) -> StateGraph:
        """
        编译工作流
        
        Returns:
            编译后的图
        """
        if self.state_graph is None:
            self.build_workflow()
        
        print("=== 编译工作流 ===")
        self.compiled_graph = self.state_graph.compile()
        print("=== 工作流编译完成 ===\n")
        
        return self.compiled_graph
    
    def get_graph(self):
        """
        获取编译后的图
        
        Returns:
            编译后的图实例
        """
        if self.compiled_graph is None:
            self.compile_workflow()
        
        return self.compiled_graph
    
    def visualize_graph(self, save_path: str = None):
        """
        可视化工作流图
        
        Args:
            save_path: 保存路径（可选）
        """
        try:
            from IPython.display import Image, display
            
            if self.compiled_graph is None:
                self.compile_workflow()
            
            # 生成图像
            graph_image = self.compiled_graph.get_graph(xray=True).draw_mermaid_png()
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(graph_image)
                print(f"工作流图已保存到: {save_path}")
            
            # 在 Jupyter 中显示
            try:
                display(Image(graph_image))
            except:
                print("无法在当前环境中显示图像")
                
        except ImportError:
            print("可视化需要额外的依赖包，跳过图像生成")
        except Exception as e:
            print(f"生成可视化图像时出错: {e}")
    
    def initialize(self):
        """
        完整初始化工作流
        
        Returns:
            编译后的图
        """
        print("=== 初始化工作流 ===")
        graph = self.compile_workflow()
        print("=== 工作流初始化完成 ===\n")
        return graph


# 创建全局工作流构建器实例
workflow_builder = WorkflowBuilder()