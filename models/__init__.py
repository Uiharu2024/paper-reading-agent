# models/__init__.py
"""
模型管理与数据结构定义模块
"""

from .llm import get_llm, LLMTaskType
from .embedding import get_embedding_model
from .schemas import (
    ReadingRequest,
    ReadingResponse,
    AnnotationRecord,
    KnowledgeEdge,
    UserProfileData
)

__all__ = [
    "get_llm",
    "LLMTaskType",
    "get_embedding_model",
    "ReadingRequest",
    "ReadingResponse",
    "AnnotationRecord",
    "KnowledgeEdge",
    "UserProfileData",
]