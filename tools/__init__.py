# tools/__init__.py
"""
论文阅读多智能体系统 - Agent 工具集合
"""

from .paper_search import search_paper_internal
from .scholar_api import search_semantic_scholar
from .arxiv_search import search_arxiv
from .web_search import search_web

# 将所有工具聚合为一个列表，方便 LangChain 绑定
ALL_TOOLS = [
    search_paper_internal,
    search_semantic_scholar,
    search_arxiv,
    search_web
]

__all__ = [
    "search_paper_internal",
    "search_semantic_scholar",
    "search_arxiv",
    "search_web",
    "ALL_TOOLS",
]