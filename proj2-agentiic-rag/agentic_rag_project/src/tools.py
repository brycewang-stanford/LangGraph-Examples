"""工具定义模块 - 定义 Agent 可使用的工具"""
## 只定义和注册了一种工具

from typing import List
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool
from .vector_store import vector_store_manager
from .config import config


class ToolsManager:
    """工具管理器类"""
    
    def __init__(self):
        """初始化工具管理器"""
        self.config = config
        self.retriever_tool = None
        self.tools = []
    
    def create_retriever_tool(self) -> Tool:
        """
        创建检索工具
        
        Returns:
            检索工具实例
        """
        if vector_store_manager.retriever is None:
            raise ValueError("检索器尚未初始化，请先调用 vector_store_manager.initialize()")
        
        print(f"正在创建检索工具: {self.config.RETRIEVER_TOOL_NAME}")
        
        self.retriever_tool = create_retriever_tool(
            vector_store_manager.retriever,
            self.config.RETRIEVER_TOOL_NAME,
            self.config.RETRIEVER_TOOL_DESC,
        )
        
        print("检索工具创建成功")
        return self.retriever_tool
    
    def get_tools(self) -> List[Tool]:
        """
        获取所有可用工具的列表
        
        Returns:
            工具列表
        """
        if not self.tools:
            # 创建检索工具
            retriever_tool = self.create_retriever_tool()
            self.tools = [retriever_tool]
            
            print(f"已注册 {len(self.tools)} 个工具:")
            for i, tool in enumerate(self.tools, 1):
                print(f"  {i}. {tool.name}: {tool.description}")
        
        return self.tools
    
    def initialize(self) -> List[Tool]:
        """
        初始化并返回所有工具
        
        Returns:
            初始化后的工具列表
        """
        print("=== 初始化工具 ===")
        tools = self.get_tools()
        print("=== 工具初始化完成 ===\n")
        return tools


# 创建全局工具管理器实例
tools_manager = ToolsManager()