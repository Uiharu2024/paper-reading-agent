# ui/__init__.py
"""
论文阅读多智能体系统 - 用户界面组件模块

提供三大核心 UI 组件:
1. ChatInterface: 对话式交互界面
2. PDFViewer: PDF 文档阅读器
3. KnowledgeGraphVisualizer: 知识图谱可视化
"""

from .chat_interface import ChatInterface
from .pdf_viewer import PDFViewer
from .knowledge_graph_viz import KnowledgeGraphVisualizer

__all__ = [
    "ChatInterface",
    "PDFViewer",
    "KnowledgeGraphVisualizer",
]