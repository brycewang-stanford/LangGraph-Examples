"""
配置模块

集中管理所有配置参数，包括 API 密钥、模型参数、URL 等。
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置 USER_AGENT 避免警告
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "adaptive-rag-system/1.0"


class Config:
    """
    配置管理类
    
    提供系统配置的集中管理。
    所有配置项都可以通过环境变量覆盖。
    """
    
    # ===== API 密钥配置 =====
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # LangSmith 配置（已禁用）
    # LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    # LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    # LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "adaptive-rag")
    
    # ===== 模型配置 =====
    # 默认模型
    DEFAULT_LLM_MODEL: str = "gpt-5"
    DEFAULT_GENERATION_MODEL: str = "gpt-5"
    DEFAULT_REWRITE_MODEL: str = "gpt-5"
    
    # 温度参数
    DEFAULT_TEMPERATURE: float = 0.0
    
    # ===== 向量存储配置 =====
    # Chroma 配置
    CHROMA_COLLECTION_NAME: str = "rag-chroma"
    
    # 文本分割参数
    CHUNK_SIZE: int = 500  # Token 数量
    CHUNK_OVERLAP: int = 0  # 重叠 Token 数量
    
    # 检索参数
    RETRIEVAL_K: int = 4  # 默认检索文档数量
    
    # ===== 默认文档源 =====
    DEFAULT_URLS: List[str] = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    # ===== 网络搜索配置 =====
    WEB_SEARCH_K: int = 3  # 网络搜索结果数量
    
    # ===== 工作流配置 =====
    MAX_ITERATIONS: int = 3  # 最大重试次数
    
    # ===== 调试配置 =====
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    VERBOSE: bool = os.getenv("VERBOSE", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """
        验证配置是否有效
        
        返回:
            配置是否有效
        """
        errors = []
        
        # 检查必需的 API 密钥
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY 未设置")
        
        if not cls.TAVILY_API_KEY:
            errors.append("TAVILY_API_KEY 未设置")
        
        # 如果有错误，打印并返回 False
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """
        打印当前配置（隐藏敏感信息）
        """
        print("\n===== 当前配置 =====")
        print(f"OpenAI API Key: {'已设置' if cls.OPENAI_API_KEY else '未设置'}")
        print(f"Tavily API Key: {'已设置' if cls.TAVILY_API_KEY else '未设置'}")
        print(f"Cohere API Key: {'已设置' if cls.COHERE_API_KEY else '未设置'}")
        # print(f"LangSmith 追踪: {'启用' if cls.LANGCHAIN_TRACING_V2 else '禁用'}")
        print(f"LangSmith 追踪: 已禁用")
        print(f"默认 LLM 模型: {cls.DEFAULT_LLM_MODEL}")
        print(f"块大小: {cls.CHUNK_SIZE} tokens")
        print(f"检索文档数: {cls.RETRIEVAL_K}")
        print(f"网络搜索结果数: {cls.WEB_SEARCH_K}")
        print(f"调试模式: {'启用' if cls.DEBUG_MODE else '禁用'}")
        print("==================\n")
    
    @classmethod
    def setup_environment(cls):
        """
        设置环境变量
        
        用于初始化系统环境。
        """
        # 设置 API 密钥到环境变量
        if cls.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        
        if cls.COHERE_API_KEY:
            os.environ["COHERE_API_KEY"] = cls.COHERE_API_KEY
        
        if cls.TAVILY_API_KEY:
            os.environ["TAVILY_API_KEY"] = cls.TAVILY_API_KEY
        
        # 设置 LangSmith（已禁用）
        # if cls.LANGCHAIN_TRACING_V2:
        #     os.environ["LANGCHAIN_TRACING_V2"] = "true"
        #     if cls.LANGCHAIN_API_KEY:
        #         os.environ["LANGCHAIN_API_KEY"] = cls.LANGCHAIN_API_KEY
        #     os.environ["LANGCHAIN_PROJECT"] = cls.LANGCHAIN_PROJECT
        
        # 显式禁用 LangSmith 追踪
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        
        
        # 禁用 LangSmith 相关警告
        import warnings
        warnings.filterwarnings("ignore", ".*LangSmith.*")


def get_config() -> Config:
    """
    获取配置实例
    
    返回:
        Config 实例
    """
    return Config()


def check_api_keys() -> bool:
    """
    检查 API 密钥是否已设置
    
    返回:
        所有必需的 API 密钥是否已设置
    """
    config = get_config()
    return config.validate()