"""
主程序入口

这是 Adaptive RAG 系统的主程序，提供命令行界面和示例用法。
"""

import sys
import argparse
from typing import Optional, List
from pprint import pprint

from config import Config, get_config
from retriever import VectorStoreManager, create_default_vectorstore
from llm_components import LLMComponents
from tools import ToolManager
from workflow import AdaptiveRAGWorkflow


def setup_system(use_default_docs: bool = True, urls: Optional[List[str]] = None):
    """
    设置系统组件
    
    参数:
        use_default_docs: 是否使用默认文档
        urls: 自定义文档 URL 列表
        
    返回:
        配置好的 AdaptiveRAGWorkflow 实例
    """
    print("\n===== 初始化 Adaptive RAG 系统 =====\n")
    
    # 1. 加载配置
    config = get_config()
    config.setup_environment()
    
    # 验证配置
    if not config.validate():
        print("错误：配置验证失败，请检查 .env 文件")
        sys.exit(1)
    
    if config.VERBOSE:
        config.print_config()
    
    # 2. 创建向量存储
    print("正在创建向量存储...")
    if use_default_docs:
        vector_store_manager = create_default_vectorstore()
    else:
        vector_store_manager = VectorStoreManager()
        if urls:
            vector_store_manager.setup_from_urls(urls)
        else:
            print("警告：未提供文档 URL，向量存储为空")
    
    # 3. 创建 LLM 组件
    print("正在初始化 LLM 组件...")
    llm_components = LLMComponents(
        model_name=config.DEFAULT_LLM_MODEL,
        temperature=config.DEFAULT_TEMPERATURE
    )
    
    # 4. 创建工具管理器
    print("正在初始化工具...")
    tool_manager = ToolManager()
    
    # 5. 创建工作流
    print("正在构建工作流...")
    workflow = AdaptiveRAGWorkflow(
        vector_store_manager=vector_store_manager,
        llm_components=llm_components,
        tool_manager=tool_manager
    )
    
    # 6. 编译工作流
    workflow.compile()
    
    print("\n系统初始化完成！\n")
    return workflow


def run_example_queries(workflow: AdaptiveRAGWorkflow):
    """
    运行示例查询
    
    参数:
        workflow: 配置好的工作流
    """
    print("\n===== 运行示例查询 =====\n")
    
    # 示例问题列表
    example_questions = [
        # 向量存储相关问题
        "What are the types of agent memory?",
        "What is prompt engineering?",
        
        # 网络搜索相关问题
        "What player at the Bears expected to draft first in the 2024 NFL draft?",
        "What's the latest news about OpenAI?",
    ]
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n示例 {i}: {question}")
        print("-" * 50)
        
        result = workflow.run(question)
        
        print(f"\n最终答案:")
        print(result.get("generation", "未能生成答案"))
        print("=" * 50)


def interactive_mode(workflow: AdaptiveRAGWorkflow):
    """
    交互式模式
    
    参数:
        workflow: 配置好的工作流
    """
    print("\n===== 交互式模式 =====")
    print("输入你的问题（输入 'quit' 或 'exit' 退出）")
    print("输入 'help' 查看帮助信息")
    print("=" * 50)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n问题> ").strip()
            
            # 检查退出命令
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            # 帮助信息
            if question.lower() == 'help':
                print_help()
                continue
            
            # 显示工作流结构
            if question.lower() == 'graph':
                print(workflow.visualize_graph())
                continue
            
            # 空输入
            if not question:
                continue
            
            # 运行查询
            print("\n处理中...")
            result = workflow.run(question)
            
            # 显示结果
            print("\n" + "=" * 50)
            print("答案:")
            print(result.get("generation", "未能生成答案"))
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def print_help():
    """
    打印帮助信息
    """
    help_text = """
    可用命令:
    - 输入任何问题进行查询
    - 'help': 显示此帮助信息
    - 'graph': 显示工作流结构
    - 'quit'/'exit'/'q': 退出程序
    
    系统功能:
    1. 向量存储查询：关于智能体、提示工程、对抗性攻击的问题
    2. 网络搜索：关于最新事件、新闻的问题
    3. 自适应路由：自动选择最佳信息源
    4. 自纠正机制：自动改进查询和答案质量
    """
    print(help_text)


def main():
    """
    主函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Adaptive RAG System - 自适应检索增强生成系统"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "example", "single"],
        default="interactive",
        help="运行模式：interactive（交互式）、example（示例）、single（单次查询）"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="单次查询模式下的问题"
    )
    
    parser.add_argument(
        "--urls",
        nargs="+",
        help="自定义文档 URL 列表"
    )
    
    parser.add_argument(
        "--no-default-docs",
        action="store_true",
        help="不使用默认文档"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置系统
    workflow = setup_system(
        use_default_docs=not args.no_default_docs,
        urls=args.urls
    )
    
    # 根据模式运行
    if args.mode == "interactive":
        interactive_mode(workflow)
    elif args.mode == "example":
        run_example_queries(workflow)
    elif args.mode == "single":
        if not args.question:
            print("错误：单次查询模式需要提供 --question 参数")
            sys.exit(1)
        
        print(f"\n问题: {args.question}")
        print("-" * 50)
        
        if args.verbose:
            result = workflow.run_with_details(args.question)
            print("\n执行轨迹:")
            pprint(result["execution_trace"])
        else:
            result = workflow.run(args.question)
        
        print(f"\n答案:")
        print(result.get("generation") or result.get("answer") or "未能生成答案")


if __name__ == "__main__":
    main()