# memory/__init__.py
"""
论文阅读多智能体系统 - 记忆管理模块

提供三层记忆架构:
1. 短期记忆: LangGraph Checkpoint (在 graph/workflow.py 中实现)
2. 长期记忆: 向量数据库 (Chroma/FAISS)
3. 结构化记忆: 知识图谱 (NetworkX)
"""

from .memory_manager import MemoryManager
from .vector_store import VectorStoreManager
from .knowledge_graph import KnowledgeGraphManager
from .user_profile import UserProfileManager

__all__ = [
    "MemoryManager",
    "VectorStoreManager",
    "KnowledgeGraphManager",
    "UserProfileManager",
]