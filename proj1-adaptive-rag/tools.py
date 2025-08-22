"""
工具模块

这个模块包含系统使用的外部工具，如网络搜索工具。
工具是 LangChain 中的重要概念，允许 LLM 与外部世界交互。
"""

from typing import Dict, List, Any
import warnings

# 抑制 TavilySearchResults 弃用警告
warnings.filterwarnings("ignore", ".*TavilySearchResults.*")

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document


class WebSearchTool:
    """
    网络搜索工具封装类
    
    封装 Tavily 搜索 API，提供网络搜索功能。
    这使得系统能够获取最新的互联网信息，补充向量存储中的静态知识。
    """
    
    def __init__(self, k: int = 3):
        """
        初始化网络搜索工具
        
        参数:
            k: 返回的搜索结果数量
        """
        self.k = k
        # Tavily 是专门为 AI 应用设计的搜索 API
        # 它返回的结果已经过优化，适合 LLM 处理
        self.search_tool = TavilySearchResults(k=k)
    
    def search(self, query: str) -> Document:
        """
        执行网络搜索并返回格式化的结果
        
        参数:
            query: 搜索查询
            
        返回:
            包含搜索结果的 Document 对象
        """
        print(f"正在搜索网络: {query}")
        
        # 调用 Tavily API 进行搜索
        results = self.search_tool.invoke({"query": query})
        
        # 格式化搜索结果
        formatted_results = self._format_results(results)
        
        # 将结果封装为 Document 对象
        # 这样可以与向量存储返回的文档格式保持一致
        web_results_doc = Document(page_content=formatted_results)
        
        print(f"找到 {len(results)} 条相关结果")
        return web_results_doc
    
    def search_raw(self, query: str) -> List[Dict[str, Any]]:
        """
        执行网络搜索并返回原始结果
        
        参数:
            query: 搜索查询
            
        返回:
            原始搜索结果列表
        """
        return self.search_tool.invoke({"query": query})
    
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化搜索结果为文本
        
        参数:
            results: 原始搜索结果
            
        返回:
            格式化的文本字符串
        """
        # 提取每个结果的内容并合并
        formatted = []
        for i, result in enumerate(results, 1):
            # 每个结果包含 content（内容）和 url（来源）
            content = result.get("content", "")
            url = result.get("url", "")
            
            # 添加结果编号和来源信息
            formatted.append(f"结果 {i}:\n{content}\n来源: {url}")
        
        # 用换行符连接所有结果
        return "\n\n".join(formatted)
    
    def update_k(self, new_k: int):
        """
        更新返回结果数量
        
        参数:
            new_k: 新的结果数量
        """
        self.k = new_k
        self.search_tool = TavilySearchResults(k=new_k)


class ToolManager:
    """
    工具管理器
    
    集中管理所有外部工具，提供统一的接口。
    未来可以扩展添加更多工具，如计算器、代码执行器等。
    """
    
    def __init__(self):
        """
        初始化工具管理器
        """
        # 初始化网络搜索工具
        self.web_search = WebSearchTool()
        
        # 未来可以添加更多工具
        # self.calculator = CalculatorTool()
        # self.code_executor = CodeExecutorTool()
        # self.database_query = DatabaseQueryTool()
    
    def get_web_search_tool(self) -> WebSearchTool:
        """
        获取网络搜索工具
        
        返回:
            WebSearchTool 实例
        """
        return self.web_search
    
    def search_web(self, query: str) -> Document:
        """
        便捷方法：执行网络搜索
        
        参数:
            query: 搜索查询
            
        返回:
            搜索结果文档
        """
        return self.web_search.search(query)
    
    def list_available_tools(self) -> List[str]:
        """
        列出所有可用的工具
        
        返回:
            工具名称列表
        """
        return ["web_search"]  # 未来可以扩展更多工具