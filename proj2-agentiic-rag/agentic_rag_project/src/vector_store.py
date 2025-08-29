"""向量存储和检索模块 - 处理文档索引和检索"""

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .config import config


class VectorStoreManager:
    """管理向量存储和检索的类"""
    
    def __init__(self):
        """初始化向量存储管理器"""
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.embeddings = OpenAIEmbeddings()
    
    def load_documents(self, urls: List[str] = None) -> List[Document]:
        """
        从 URL 加载文档
        
        Args:
            urls: 要加载的 URL 列表，默认使用配置中的 URL
            
        Returns:
            加载的文档列表
        """
        if urls is None:
            urls = self.config.BLOG_URLS
        
        print(f"正在从 {len(urls)} 个 URL 加载文档...")
        docs = []
        for url in urls:
            print(f"  - 加载: {url}")
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        
        print(f"成功加载 {len(docs)} 个文档")
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成更小的块
        
        Args:
            documents: 要分割的文档列表
            
        Returns:
            分割后的文档块列表
        """
        print(f"正在分割文档 (chunk_size={self.config.CHUNK_SIZE}, overlap={self.config.CHUNK_OVERLAP})...")
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        
        doc_splits = text_splitter.split_documents(documents)
        print(f"文档已分割成 {len(doc_splits)} 个块")
        
        return doc_splits
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        创建向量存储
        
        Args:
            documents: 要索引的文档列表
            
        Returns:
            创建的向量存储实例
        """
        print(f"正在创建向量存储 (collection={self.config.COLLECTION_NAME})...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            collection_name=self.config.COLLECTION_NAME,
            embedding=self.embeddings,
        )
        
        print("向量存储创建成功")
        return self.vectorstore
    
    def get_retriever(self):
        """
        获取检索器
        
        Returns:
            检索器实例
        """
        if self.vectorstore is None:
            raise ValueError("向量存储尚未创建，请先调用 create_vectorstore()")
        
        if self.retriever is None:
            self.retriever = self.vectorstore.as_retriever()
            print("检索器已创建")
        
        return self.retriever
    
    def initialize(self) -> None:
        """
        完整的初始化流程：加载、分割、索引文档
        """
        print("=== 初始化向量存储 ===")
        
        # 1. 加载文档
        documents = self.load_documents()
        
        # 2. 分割文档
        doc_splits = self.split_documents(documents)
        
        # 3. 创建向量存储
        self.create_vectorstore(doc_splits)
        
        # 4. 创建检索器
        self.get_retriever()
        
        print("=== 向量存储初始化完成 ===\n")
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        执行相似度搜索
        
        Args:
            query: 搜索查询
            k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if self.retriever is None:
            raise ValueError("检索器尚未创建，请先调用 initialize()")
        
        return self.retriever.get_relevant_documents(query)


# 创建全局实例
vector_store_manager = VectorStoreManager()