"""
数据模型模块

这个模块定义了 Adaptive RAG 系统中使用的所有数据模型。
使用 Pydantic 来确保数据验证和类型安全。
"""

from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """
    路由查询模型
    
    用于决定将用户问题路由到向量存储还是网络搜索。
    这是 LangGraph 中的关键决策点，体现了自适应 RAG 的核心思想。
    """
    
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题选择路由到网络搜索还是向量存储"
    )


class GradeDocuments(BaseModel):
    """
    文档相关性评分模型
    
    用于评估检索到的文档是否与用户问题相关。
    这是自纠正 RAG 的重要组成部分，确保只使用相关文档生成答案。
    """
    
    binary_score: str = Field(
        description="文档是否与问题相关，'yes' 或 'no'"
    )


class GradeHallucinations(BaseModel):
    """
    幻觉检测模型
    
    用于评估生成的答案是否基于检索到的事实。
    防止模型产生幻觉（虚假信息）是 RAG 系统的关键质量控制点。
    """
    
    binary_score: str = Field(
        description="答案是否基于事实，'yes' 或 'no'"
    )


class GradeAnswer(BaseModel):
    """
    答案质量评分模型
    
    用于评估生成的答案是否真正回答了用户的问题。
    这是质量控制的最后一道关卡。
    """
    
    binary_score: str = Field(
        description="答案是否解决了问题，'yes' 或 'no'"
    )


class GraphState(TypedDict):
    """
    LangGraph 图状态模型
    
    这是 LangGraph 的核心概念之一：图状态（Graph State）。
    它定义了在整个工作流中传递的数据结构。
    
    LangGraph 知识点：
    - GraphState 在每个节点之间传递，保持工作流的状态
    - 每个节点可以读取和修改状态中的字段
    - 状态的不可变性确保了工作流的可追踪性和可调试性
    
    属性:
        question: 用户的原始问题或重写后的问题
        generation: LLM 生成的答案
        documents: 检索到的文档列表或网络搜索结果
    """
    
    question: str  # 当前的问题（可能被重写过）
    generation: str  # LLM 生成的答案
    documents: List[str]  # 相关文档列表