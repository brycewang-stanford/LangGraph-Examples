#!/usr/bin/env python3
"""
Agentic RAG 系统启动脚本
使用方法:
    python run.py "你的问题"  # 命令行模式
    python run.py            # 交互模式
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 现在可以导入 src 模块
from src.main import main

if __name__ == "__main__":
    main()