"""
检索和索引模块

这个模块负责创建和管理向量存储索引，以及文档检索功能。
包括文档加载、分割、向量化和检索等功能。
"""

from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorStoreManager:
    """
    向量存储管理器
    
    负责创建、管理和查询向量存储。
    这是 RAG 系统的核心组件之一，提供知识库的存储和检索能力。
    """
    
    def __init__(
        self,
        embedding_model: Optional[OpenAIEmbeddings] = None,
        collection_name: str = "rag-chroma",
        chunk_size: int = 500,
        chunk_overlap: int = 0
    ):
        """
        初始化向量存储管理器
        
        参数:
            embedding_model: 嵌入模型，如果为 None 则创建默认的 OpenAI 嵌入
            collection_name: Chroma 集合名称
            chunk_size: 文本分块大小（以 token 为单位）
            chunk_overlap: 分块之间的重叠大小
        """
        # 如果没有提供嵌入模型，创建默认的 OpenAI 嵌入
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.retriever = None
        
    def load_documents_from_urls(self, urls: List[str]) -> List[Document]:
        """
        从 URL 列表加载文档
        
        参数:
            urls: 要加载的网页 URL 列表
            
        返回:
            加载的文档列表
        """
        print(f"正在从 {len(urls)} 个 URL 加载文档...")
        
        # 并行加载所有 URL 的内容
        docs = [WebBaseLoader(url).load() for url in urls]
        
        # 将嵌套列表展平为单一列表
        docs_list = [item for sublist in docs for item in sublist]
        
        print(f"成功加载 {len(docs_list)} 个文档")
        return docs_list
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成更小的块
        
        使用 RecursiveCharacterTextSplitter 进行智能分割，
        它会尝试在自然断点（如段落、句子）处分割文本。
        
        参数:
            documents: 要分割的文档列表
            
        返回:
            分割后的文档块列表
        """
        print(f"正在分割 {len(documents)} 个文档...")
        
        # 使用 tiktoken 编码器创建文本分割器
        # tiktoken 是 OpenAI 的分词器，能准确计算 token 数量
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # 分割所有文档
        doc_splits = text_splitter.split_documents(documents)
        
        print(f"文档被分割成 {len(doc_splits)} 个块")
        return doc_splits
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        创建向量存储
        
        将文档转换为向量并存储在 Chroma 数据库中。
        
        参数:
            documents: 要存储的文档列表
            
        返回:
            Chroma 向量存储实例
        """
        print("正在创建向量存储...")
        
        # 使用 Chroma 创建向量存储
        # Chroma 是一个轻量级的向量数据库，适合开发和小规模应用
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        
        # 创建检索器接口
        self.retriever = self.vectorstore.as_retriever()
        
        print(f"向量存储创建完成，包含 {len(documents)} 个文档块")
        return self.vectorstore
    
    def setup_from_urls(self, urls: List[str]):
        """
        从 URL 列表完整设置向量存储
        
        这是一个便捷方法，执行完整的索引创建流程。
        
        参数:
            urls: 要索引的网页 URL 列表
        """
        # 1. 加载文档
        documents = self.load_documents_from_urls(urls)
        
        # 2. 分割文档
        doc_splits = self.split_documents(documents)
        
        # 3. 创建向量存储
        self.create_vectorstore(doc_splits)
        
        print("向量存储设置完成！")
    
    def get_retriever(self):
        """
        获取检索器
        
        返回:
            配置好的检索器，如果向量存储未初始化则返回 None
        """
        if self.retriever is None:
            print("警告：检索器尚未初始化，请先创建向量存储")
        return self.retriever
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        执行相似度搜索
        
        参数:
            query: 搜索查询
            k: 返回的文档数量
            
        返回:
            最相关的文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储尚未初始化")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        执行带评分的相似度搜索
        
        参数:
            query: 搜索查询
            k: 返回的文档数量
            
        返回:
            (文档, 相似度分数) 元组列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储尚未初始化")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)


def create_default_vectorstore() -> VectorStoreManager:
    """
    创建默认配置的向量存储
    
    使用预定义的 URL 创建一个包含 AI 相关文档的向量存储。
    
    返回:
        配置好的 VectorStoreManager 实例
    """
    # 默认的文档 URL
    # 这些是 Lilian Weng 的博客文章，涵盖了智能体、提示工程和对抗性攻击等主题
    default_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",  # 关于 AI 智能体
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",  # 提示工程
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",  # 对抗性攻击
    ]
    
    # 创建向量存储管理器
    manager = VectorStoreManager()
    
    # 设置向量存储
    manager.setup_from_urls(default_urls)
    
    return manager