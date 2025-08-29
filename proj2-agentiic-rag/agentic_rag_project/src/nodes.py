"""节点函数模块 - 定义工作流中的所有节点"""

from typing import Literal
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from .config import config
from .tools import tools_manager


class Grade(BaseModel):
    """文档相关性评分模型"""
    binary_score: str = Field(description="相关性评分 'yes' 或 'no'")


class AgentNodes:
    """包含所有节点函数的类"""
    
    def __init__(self):
        """初始化节点"""
        self.config = config
        self.tools = None  # 延迟初始化
    
    def get_tools(self):
        """获取工具列表（延迟初始化）"""
        if self.tools is None:
            self.tools = tools_manager.get_tools()
        return self.tools
    
    def agent(self, state):
        """
        代理节点：决定是否需要调用检索工具
        
        Args:
            state: 当前状态，包含消息列表
            
        Returns:
            dict: 更新后的状态
        """
        print("---调用代理节点---")
        messages = state["messages"]
        
        # 创建绑定了工具的模型
        model = ChatOpenAI(
            temperature=self.config.DEFAULT_TEMPERATURE,
            streaming=self.config.STREAMING,
            model=self.config.AGENT_MODEL  # 使用 gpt-5-mini
        )
        model = model.bind_tools(self.get_tools())
        
        # 调用模型
        response = model.invoke(messages)
        
        # 返回更新后的消息列表
        return {"messages": [response]}
    
    def grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        文档相关性评估节点
        
        Args:
            state: 当前状态
            
        Returns:
            str: 决策结果 ("generate" 或 "rewrite")
        """
        print("---检查文档相关性---")
        
        # 创建评分模型
        model = ChatOpenAI(
            temperature=self.config.DEFAULT_TEMPERATURE,
            model=self.config.GRADER_MODEL,  # 使用 gpt-5
            streaming=self.config.STREAMING
        )
        
        # 绑定结构化输出
        llm_with_tool = model.with_structured_output(Grade)
        
        # 创建评估提示
        prompt = PromptTemplate(
            template="""你是一个评估检索文档与用户问题相关性的评分员。
            以下是检索到的文档: 

            {context} 

            以下是用户问题: {question} 
            
            如果文档包含与用户问题相关的关键词或语义信息，则评定为相关。
            请给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。""",
            input_variables=["context", "question"],
        )
        
        # 创建评估链
        chain = prompt | llm_with_tool
        
        # 获取问题和文档
        messages = state["messages"]
        last_message = messages[-1]
        
        question = messages[0].content
        docs = last_message.content
        
        # 执行评估
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
        
        if score == "yes":
            print("---决策：文档相关---")
            return "generate"
        else:
            print("---决策：文档不相关---")
            print(f"评分: {score}")
            return "rewrite"
    
    def rewrite(self, state):
        """
        查询重写节点
        
        Args:
            state: 当前状态
            
        Returns:
            dict: 包含重写后问题的更新状态
        """
        print("---重写查询---")
        messages = state["messages"]
        question = messages[0].content
        
        # 创建重写消息
        msg = [
            HumanMessage(
                content=f"""
                请分析输入内容并尝试理解其潜在的语义意图和含义。
                
                以下是原始问题:
                -------
                {question}
                -------
                
                请重新表述一个改进的问题:
                """,
            )
        ]
        
        # 创建重写模型
        model = ChatOpenAI(
            temperature=self.config.DEFAULT_TEMPERATURE,
            model=self.config.REWRITER_MODEL,  # 使用 gpt-5
            streaming=self.config.STREAMING
        )
        
        # 执行重写
        response = model.invoke(msg)
        return {"messages": [response]}
    
    def generate(self, state):
        """
        答案生成节点
        
        Args:
            state: 当前状态
            
        Returns:
            dict: 包含生成答案的更新状态
        """
        print("---生成答案---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        
        docs = last_message.content
        
        # 从 hub 拉取 RAG 提示模板
        prompt = hub.pull(self.config.RAG_PROMPT_HUB)
        
        # 创建生成模型
        llm = ChatOpenAI(
            model_name=self.config.GENERATOR_MODEL,  # 使用 gpt-5
            temperature=self.config.DEFAULT_TEMPERATURE,
            streaming=self.config.STREAMING
        )
        
        # 创建 RAG 链
        rag_chain = prompt | llm | StrOutputParser()
        
        # 生成答案
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}


# 创建全局节点实例
agent_nodes = AgentNodes()