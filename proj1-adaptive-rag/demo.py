"""
演示程序

展示 Adaptive RAG 系统的工作流程和架构，不需要 API 密钥。
"""

import sys
from workflow import AdaptiveRAGWorkflow


def demo_workflow_structure():
    """
    演示工作流结构
    """
    print("\n" + "="*60)
    print("Adaptive RAG 系统架构演示")
    print("="*60)
    
    print("\n### 系统组件：")
    print("""
    1. 数据模型 (models.py)
       - GraphState: LangGraph 状态管理
       - RouteQuery: 路由决策模型
       - GradeDocuments: 文档评分模型
       - GradeHallucinations: 幻觉检测模型
       - GradeAnswer: 答案质量评分模型
    
    2. LLM 组件 (llm_components.py)
       - 问题路由器
       - 文档评分器
       - RAG 生成链
       - 幻觉检测器
       - 答案评分器
       - 问题重写器
    
    3. 检索系统 (retriever.py)
       - 向量存储管理
       - 文档分割
       - 相似度搜索
    
    4. 工具 (tools.py)
       - 网络搜索工具
       - 工具管理器
    
    5. 图节点 (graph_nodes.py)
       - retrieve: 检索文档
       - web_search: 网络搜索
       - grade_documents: 文档评分
       - generate: 生成答案
       - transform_query: 查询重写
    
    6. 图边缘 (graph_edges.py)
       - route_question: 路由决策
       - decide_to_generate: 生成决策
       - grade_generation: 质量评估
    """)
    
    print("\n### LangGraph 工作流：")
    print("""
    START
      |
      v
    [路由问题] ← 决定使用哪个数据源
      /     \\
     /       \\
    v         v
[网络搜索]  [检索文档]
    |         |
    |         v
    |    [评分文档] ← 过滤不相关文档
    |       /    \\
    |      /      \\
    |     v        v
    |  [转换查询]  |
    |     |        |
    |     v        |
    |  [检索文档]  |
    |              |
    v              v
    [====生成答案====]
            |
            v
      [质量控制] ← 检查幻觉和答案质量
       /   |   \\
      /    |    \\
     v     v     v
  [结束] [重试] [转换]
    """)
    
    print("\n### LangGraph 核心概念：")
    print("""
    1. **状态 (State)**
       - 在节点间传递的数据结构
       - 包含: question, documents, generation
    
    2. **节点 (Nodes)**
       - 执行具体任务的函数
       - 接收状态，返回更新后的状态
    
    3. **边缘 (Edges)**
       - 连接节点的路径
       - 条件边缘根据状态决定下一步
    
    4. **编译 (Compile)**
       - 将图结构转换为可执行应用
       - 支持 stream() 和 invoke() 方法
    """)
    
    print("\n### 自适应特性：")
    print("""
    1. **智能路由**
       - 自动选择向量存储或网络搜索
       - 基于问题内容做决策
    
    2. **自纠正机制**
       - 文档相关性过滤
       - 查询重写优化
       - 答案质量控制
    
    3. **质量保证**
       - 幻觉检测
       - 答案完整性检查
       - 多次重试机制
    """)


def demo_execution_flow():
    """
    演示执行流程
    """
    print("\n" + "="*60)
    print("执行流程演示")
    print("="*60)
    
    # 示例1：向量存储路径
    print("\n### 示例1：知识库问题")
    print("问题: 'What are the types of agent memory?'")
    print("\n执行流程:")
    print("1. START → 路由问题")
    print("   决策: 使用向量存储（问题与知识库主题相关）")
    print("2. → 检索文档")
    print("   从向量存储中检索相关文档")
    print("3. → 评分文档")
    print("   过滤不相关文档，保留3个相关文档")
    print("4. → 生成答案")
    print("   基于文档生成答案")
    print("5. → 质量控制")
    print("   检查：无幻觉 ✓")
    print("   检查：答案完整 ✓")
    print("6. → END")
    print("\n答案: 智能体记忆类型包括感觉记忆、短期记忆和长期记忆...")
    
    # 示例2：网络搜索路径
    print("\n### 示例2：实时信息问题")
    print("问题: 'What's the latest news about OpenAI?'")
    print("\n执行流程:")
    print("1. START → 路由问题")
    print("   决策: 使用网络搜索（需要最新信息）")
    print("2. → 网络搜索")
    print("   调用 Tavily API 搜索最新信息")
    print("3. → 生成答案")
    print("   基于搜索结果生成答案")
    print("4. → 质量控制")
    print("   检查：基于事实 ✓")
    print("   检查：回答问题 ✓")
    print("5. → END")
    print("\n答案: 根据最新消息，OpenAI...")
    
    # 示例3：自纠正路径
    print("\n### 示例3：需要查询重写的问题")
    print("问题: 'Tell me about transformers'")
    print("\n执行流程:")
    print("1. START → 路由问题")
    print("   决策: 使用向量存储")
    print("2. → 检索文档")
    print("   检索到一些文档")
    print("3. → 评分文档")
    print("   所有文档不相关（关于电力变压器）")
    print("4. → 转换查询")
    print("   重写: 'transformer architecture in deep learning'")
    print("5. → 检索文档")
    print("   重新检索，找到相关文档")
    print("6. → 评分文档")
    print("   2个文档相关")
    print("7. → 生成答案")
    print("8. → 质量控制 → END")
    print("\n答案: Transformer 是一种深度学习架构...")


def main():
    """
    主函数
    """
    print("\n" + "="*60)
    print("欢迎使用 Adaptive RAG 系统演示")
    print("="*60)
    print("\n这是一个基于 LangGraph 的自适应检索增强生成系统")
    print("由于需要 API 密钥才能实际运行，这里展示系统架构和工作流程")
    
    while True:
        print("\n请选择演示内容：")
        print("1. 查看系统架构")
        print("2. 查看执行流程示例")
        print("3. 查看 LangGraph 工作流可视化")
        print("4. 退出")
        
        choice = input("\n选择 (1-4): ").strip()
        
        if choice == "1":
            demo_workflow_structure()
        elif choice == "2":
            demo_execution_flow()
        elif choice == "3":
            # 创建一个模拟的工作流对象来展示结构
            workflow = AdaptiveRAGWorkflow()
            print(workflow.visualize_graph())
        elif choice == "4":
            print("\n再见！")
            break
        else:
            print("无效选择，请重试")
        
        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()