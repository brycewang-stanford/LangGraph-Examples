"""
图边缘路由模块

这个模块定义了 LangGraph 工作流中的边缘函数（条件路由）。
边缘函数决定工作流的下一步走向。

LangGraph 核心概念：
- 边缘（Edge）：连接节点的路径
- 条件边缘（Conditional Edge）：根据状态决定下一个节点
- 边缘函数返回下一个节点的名称
"""

from typing import Literal
from pprint import pprint

from models import GraphState
from llm_components import LLMComponents


class GraphEdges:
    """
    图边缘管理类
    
    封装所有边缘函数，控制工作流的路由逻辑。
    这是实现自适应和自纠正 RAG 的关键部分。
    """
    
    def __init__(self, llm_components: LLMComponents):
        """
        初始化图边缘
        
        参数:
            llm_components: LLM 组件管理器
        """
        self.llm_components = llm_components
        
        # 创建所需的 LLM 链
        self.question_router = llm_components.create_question_router()
        self.hallucination_grader = llm_components.create_hallucination_grader()
        self.answer_grader = llm_components.create_answer_grader()
    
    def route_question(self, state: GraphState) -> Literal["vectorstore", "web_search"]:
        """
        问题路由边缘函数
        
        决定将问题路由到向量存储还是网络搜索。
        这是 Adaptive RAG 的核心决策点。
        
        LangGraph 知识点：
        - 这是一个条件边缘函数
        - 返回值决定下一个要执行的节点
        - 实现了动态路由逻辑
        
        参数:
            state: 当前图状态
            
        返回:
            下一个节点的名称："vectorstore" 或 "web_search"
        """
        print("---路由问题---")
        question = state["question"]
        
        # 使用 LLM 决定路由
        source = self.question_router.invoke({"question": question})
        
        if source.datasource == "web_search":
            print("---路由决策：网络搜索---")
            print("原因：问题需要最新信息或不在知识库范围内")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---路由决策：向量存储---")
            print("原因：问题与知识库主题相关")
            return "vectorstore"
    
    def decide_to_generate(self, state: GraphState) -> Literal["transform_query", "generate"]:
        """
        决定是否生成答案的边缘函数
        
        基于文档相关性评分结果，决定是生成答案还是重写查询。
        这实现了自纠正机制。
        
        LangGraph 知识点：
        - 边缘函数可以基于状态中的任何信息做决策
        - 这种条件逻辑实现了工作流的自适应性
        
        参数:
            state: 当前图状态
            
        返回:
            下一个节点的名称："generate" 或 "transform_query"
        """
        print("---评估文档质量---")
        filtered_documents = state.get("documents", [])
        
        if not filtered_documents:
            # 所有文档都被过滤掉了，需要重写查询
            print("---决策：所有文档都不相关，转换查询---")
            print("原因：检索到的文档与问题不匹配")
            return "transform_query"
        else:
            # 有相关文档，可以生成答案
            print("---决策：生成答案---")
            print(f"原因：找到 {len(filtered_documents)} 个相关文档")
            return "generate"
    
    def grade_generation_v_documents_and_question(
        self, 
        state: GraphState
    ) -> Literal["useful", "not useful", "not supported"]:
        """
        评估生成质量的边缘函数
        
        检查生成的答案是否：
        1. 基于事实（无幻觉）
        2. 回答了用户问题
        
        这是质量控制的最后一道关卡，确保输出的可靠性。
        
        LangGraph 知识点：
        - 边缘函数可以返回多个可能的路径
        - 这种多路径决策实现了复杂的控制流
        
        参数:
            state: 当前图状态
            
        返回:
            评估结果："useful"、"not useful" 或 "not supported"
        """
        print("---检查幻觉---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # 第一步：检查是否有幻觉
        score = self.hallucination_grader.invoke({
            "documents": documents,
            "generation": generation
        })
        grade = score.binary_score
        
        if grade == "yes":
            print("---决策：生成基于事实---")
            
            # 第二步：检查是否回答了问题
            print("---评估答案质量---")
            score = self.answer_grader.invoke({
                "question": question,
                "generation": generation
            })
            grade = score.binary_score
            
            if grade == "yes":
                print("---决策：答案有用---")
                print("原因：答案基于事实且回答了问题")
                return "useful"
            else:
                print("---决策：答案未充分回答问题---")
                print("原因：虽然基于事实，但未直接回答用户问题")
                return "not useful"
        else:
            print("---决策：生成包含幻觉，需要重试---")
            print("原因：答案包含文档中不存在的信息")
            return "not supported"
    
    def should_continue_retrieval(
        self, 
        state: GraphState, 
        max_iterations: int = 3
    ) -> Literal["continue", "stop"]:
        """
        决定是否继续检索的边缘函数
        
        用于防止无限循环，限制重试次数。
        
        参数:
            state: 当前图状态
            max_iterations: 最大迭代次数
            
        返回:
            "continue" 或 "stop"
        """
        # 这是一个扩展功能，可以跟踪迭代次数
        # 在实际实现中，需要在状态中添加迭代计数器
        iterations = state.get("iterations", 0)
        
        if iterations >= max_iterations:
            print(f"---达到最大迭代次数 ({max_iterations})，停止---")
            return "stop"
        else:
            print(f"---继续迭代 (当前: {iterations}/{max_iterations})---")
            return "continue"
    
    def debug_routing_decision(
        self, 
        state: GraphState, 
        decision: str, 
        reason: str = ""
    ):
        """
        调试辅助函数
        
        记录路由决策，用于调试和监控。
        
        参数:
            state: 当前图状态
            decision: 做出的决策
            reason: 决策原因
        """
        print("\n=== 路由决策 ===")
        print(f"当前问题: {state.get('question', 'N/A')}")
        print(f"决策: {decision}")
        if reason:
            print(f"原因: {reason}")
        print("===============\n")