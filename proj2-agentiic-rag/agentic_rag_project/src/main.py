"""主程序入口 - Agentic RAG 系统"""

import os
import sys
import pprint
from typing import Dict, Any
from .config import config
from .vector_store import vector_store_manager
from .tools import tools_manager
from .workflow import workflow_builder


class AgenticRAGSystem:
    """Agentic RAG 系统主类"""
    
    def __init__(self):
        """初始化系统"""
        self.config = config
        self.vector_store = vector_store_manager
        self.tools = tools_manager
        self.workflow_builder = workflow_builder
        self.graph = None
        self.initialized = False
    
    def initialize(self):
        """完整系统初始化"""
        print("=" * 60)
        print("🚀 Agentic RAG 系统启动")
        print("=" * 60)
        
        try:
            # 1. 验证配置
            print("1. 验证系统配置...")
            self.config.validate()
            print("   ✓ 配置验证通过")
            
            # 2. 初始化向量存储
            print("2. 初始化向量存储...")
            self.vector_store.initialize()
            print("   ✓ 向量存储初始化完成")
            
            # 3. 初始化工具
            print("3. 初始化工具...")
            self.tools.initialize()
            print("   ✓ 工具初始化完成")
            
            # 4. 构建工作流
            print("4. 构建工作流...")
            self.graph = self.workflow_builder.initialize()
            print("   ✓ 工作流构建完成")
            
            self.initialized = True
            print("\n" + "=" * 60)
            print("🎉 系统初始化完成！")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 系统初始化失败: {e}")
            sys.exit(1)
    
    def query(self, question: str, verbose: bool = True) -> str:
        """
        处理用户查询
        
        Args:
            question: 用户问题
            verbose: 是否显示详细过程
            
        Returns:
            最终答案
        """
        if not self.initialized:
            raise ValueError("系统尚未初始化，请先调用 initialize() 方法")
        
        if verbose:
            print("\n" + "=" * 50)
            print(f"🔍 处理查询: {question}")
            print("=" * 50)
        
        # 构建输入
        inputs = {
            "messages": [
                ("user", question),
            ]
        }
        
        final_answer = None
        
        # 执行工作流
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                if verbose:
                    print(f"\n📍 节点输出 '{key}':")
                    print("-" * 30)
                    pprint.pprint(value, indent=2, width=80, depth=None)
                
                # 保存最终答案（严格按照原始 notebook 逻辑）
                if key == "generate" and "messages" in value:
                    final_answer = value["messages"][0]
        
        if verbose:
            print("\n" + "=" * 50)
            print("✅ 查询处理完成")
            print("=" * 50)
        
        return final_answer if final_answer else "未能生成答案"
    
    def interactive_mode(self):
        """交互式问答模式"""
        print("\n🤖 进入交互式问答模式")
        print("输入 'quit' 或 'exit' 退出")
        print("-" * 40)
        
        while True:
            try:
                question = input("\n❓ 请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not question:
                    print("⚠️ 请输入有效问题")
                    continue
                
                # 处理查询
                answer = self.query(question, verbose=True)
                
                print(f"\n🎯 最终答案:")
                print("-" * 20)
                print(answer)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理查询时出错: {e}")
    
    def batch_query(self, questions: list, verbose: bool = False) -> Dict[str, str]:
        """
        批量处理查询
        
        Args:
            questions: 问题列表
            verbose: 是否显示详细过程
            
        Returns:
            问题答案映射
        """
        results = {}
        
        print(f"\n📋 开始批量处理 {len(questions)} 个问题...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] 处理问题: {question}")
            try:
                answer = self.query(question, verbose=verbose)
                results[question] = answer
                print(f"✓ 完成")
            except Exception as e:
                print(f"❌ 失败: {e}")
                results[question] = f"处理失败: {e}"
        
        print(f"\n📊 批量处理完成，成功处理 {len([r for r in results.values() if not r.startswith('处理失败')])} 个问题")
        return results


def main():
    """主函数"""
    # 创建系统实例
    rag_system = AgenticRAGSystem()
    
    # 初始化系统
    rag_system.initialize()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        question = " ".join(sys.argv[1:])
        print(f"\n🔍 命令行查询: {question}")
        answer = rag_system.query(question)
        print(f"\n🎯 答案: {answer}")
    else:
        # 交互式模式
        rag_system.interactive_mode()


if __name__ == "__main__":
    main()