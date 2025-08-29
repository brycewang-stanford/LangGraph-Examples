"""配置管理模块 - 集中管理所有系统配置"""

import os
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class Config:
    """系统配置类"""
    
    # OpenAI API 配置 - 统一使用 gpt-5 模型
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = "gpt-5-mini"  # 使用 gpt-555-mini 模型

    # 模型配置 - 所有模型统一使用 gpt-5-mini
    AGENT_MODEL: str = "gpt-5-mini"
    GRADER_MODEL: str = "gpt-5-mini"
    REWRITER_MODEL: str = "gpt-5-mini"
    GENERATOR_MODEL: str = "gpt-5-mini"

    # 温度参数
    DEFAULT_TEMPERATURE: float = 0.0
    
    # 流式输出
    STREAMING: bool = False
    
    # 文档源配置
    BLOG_URLS: List[str] = None
    
    # 文本分割参数
    CHUNK_SIZE: int = 100
    CHUNK_OVERLAP: int = 50
    
    # 向量数据库配置
    COLLECTION_NAME: str = "rag-chroma"
    
    # 检索工具配置
    RETRIEVER_TOOL_NAME: str = "retrieve_blog_posts"
    RETRIEVER_TOOL_DESC: str = (
        "Search and return information about Lilian Weng blog posts on "
        "LLM agents, prompt engineering, and adversarial attacks on LLMs."
    )
    
    # RAG 提示模板
    RAG_PROMPT_HUB: str = "rlm/rag-prompt"
    
    def __post_init__(self):
        """初始化后设置默认值"""
        if self.BLOG_URLS is None:
            self.BLOG_URLS = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            ]
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置实例"""
        return cls()
    
    def validate(self):
        """验证配置的有效性"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 环境变量未设置")
        return True


# 创建全局配置实例
config = Config.from_env()