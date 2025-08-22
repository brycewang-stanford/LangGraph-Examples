"""
LLM 组件模块

这个模块包含所有与大语言模型相关的组件，包括：
- 问题路由器
- 文档相关性评分器
- 答案生成器
- 幻觉检测器
- 答案质量评分器
- 问题重写器
"""

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from models import RouteQuery, GradeDocuments, GradeHallucinations, GradeAnswer


class LLMComponents:
    """
    LLM 组件管理类
    
    集中管理所有 LLM 相关的组件，提供统一的接口。
    这种设计模式使得组件易于维护和测试。
    """
    
    def __init__(self, model_name: str = "gpt-5", temperature: float = 0):
        """
        初始化 LLM 组件
        
        参数:
            model_name: OpenAI 模型名称
            temperature: 温度参数，控制生成的随机性（0 表示确定性输出）
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
    def create_question_router(self):
        """
        创建问题路由器
        
        这是 Adaptive RAG 的核心组件之一。
        根据问题内容决定使用向量存储（本地知识库）还是网络搜索（最新信息）。
        
        返回:
            可以将问题路由到不同数据源的链
        """
        # 使用结构化输出确保返回格式正确
        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        
        # 系统提示词：定义路由规则
        system = """你是一个专门的问题路由专家。
        向量存储包含关于以下主题的文档：智能体（agents）、提示工程（prompt engineering）和对抗性攻击（adversarial attacks）。
        对于这些主题的问题，使用向量存储。
        对于其他问题，特别是关于最新事件、新闻或需要实时信息的问题，使用网络搜索。"""
        
        # 构建提示模板
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])
        
        # 返回完整的链
        return route_prompt | structured_llm_router
    
    def create_retrieval_grader(self):
        """
        创建文档相关性评分器
        
        用于评估检索到的文档是否与问题相关。
        这是自纠正 RAG 的关键组件，确保只使用相关文档。
        
        返回:
            文档相关性评分链
        """
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        
        system = """你是一个评估检索文档相关性的评分专家。
        如果文档包含与用户问题相关的关键词或语义信息，将其评为相关。
        评判标准不需要过于严格，目标是过滤掉明显不相关的检索结果。
        给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。"""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "检索到的文档: \n\n {document} \n\n 用户问题: {question}"),
        ])
        
        return grade_prompt | structured_llm_grader
    
    def create_rag_chain(self):
        """
        创建 RAG 生成链
        
        基于检索到的文档生成答案。
        使用 LangChain Hub 中的标准 RAG 提示模板。
        
        返回:
            RAG 生成链
        """
        # 从 LangChain Hub 拉取经过优化的 RAG 提示模板
        prompt = hub.pull("rlm/rag-prompt")
        
        # 使用较低成本的模型进行生成
        llm = ChatOpenAI(model_name="gpt-5", temperature=0)
        
        # 构建链：提示 -> LLM -> 字符串输出
        return prompt | llm | StrOutputParser()
    
    def create_hallucination_grader(self):
        """
        创建幻觉检测器
        
        评估生成的答案是否基于提供的文档事实。
        这是确保答案准确性的重要质量控制步骤。
        
        返回:
            幻觉检测链
        """
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        
        system = """你是一个评估 LLM 生成内容是否基于事实的专家。
        判断生成的答案是否有充分的事实依据。
        给出二元评分 'yes' 或 'no'。
        'yes' 表示答案基于提供的事实，'no' 表示答案包含未经证实的信息。"""
        
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "事实集合: \n\n {documents} \n\n LLM 生成: {generation}"),
        ])
        
        return hallucination_prompt | structured_llm_grader
    
    def create_answer_grader(self):
        """
        创建答案质量评分器
        
        评估生成的答案是否真正回答了用户的问题。
        即使答案基于事实，也可能没有直接回答问题。
        
        返回:
            答案质量评分链
        """
        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)
        
        system = """你是一个评估答案是否解决问题的专家。
        判断生成的答案是否真正回答了用户的问题。
        给出二元评分 'yes' 或 'no'。
        'yes' 表示答案解决了问题，'no' 表示答案未能充分回答问题。"""
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "用户问题: \n\n {question} \n\n LLM 生成: {generation}"),
        ])
        
        return answer_prompt | structured_llm_grader
    
    def create_question_rewriter(self):
        """
        创建问题重写器
        
        当初始检索失败或答案质量不佳时，重写问题以改善检索效果。
        这是自适应 RAG 的关键特性之一。
        
        返回:
            问题重写链
        """
        # 使用稍微不同的模型版本，可能有更好的重写能力
        llm = ChatOpenAI(model="gpt-5", temperature=0)
        
        system = """你是一个问题重写专家，能够将输入问题转换为更适合向量存储检索的版本。
        分析输入问题的潜在语义意图和含义。
        重写问题时要：
        1. 保持原始意图不变
        2. 使用更规范的术语
        3. 补充可能的同义词或相关概念
        4. 使问题更加具体和清晰"""
        
        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "原始问题: \n\n {question} \n 请提供改进后的问题。"),
        ])
        
        return re_write_prompt | llm | StrOutputParser()
    
    @staticmethod
    def format_docs(docs):
        """
        格式化文档列表
        
        将文档对象列表转换为字符串格式，便于传递给 LLM。
        
        参数:
            docs: 文档对象列表
            
        返回:
            格式化的文档字符串
        """
        return "\n\n".join(doc.page_content for doc in docs)