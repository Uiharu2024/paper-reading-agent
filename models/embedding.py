# models/embedding.py
"""
Embedding 模型统一管理 (RTX 5060 8GB 优化版)
核心策略: 强制 CPU + ONNX 加速，将 8GB 显存 100% 留给 LLM
"""

import os
from typing import Optional
from functools import lru_cache
from langchain_core.embeddings import Embeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


@lru_cache(maxsize=1)
def get_embedding_model(
        model_name: Optional[str] = None,
        use_api: bool = False
) -> Embeddings:
    """
    获取 Embedding 模型实例 (全局单例)

    Args:
        model_name: 模型名称或本地路径。默认 'BAAI/bge-m3'。
        use_api: 是否使用 OpenAI API 兼容模式。

    Returns:
        Embeddings 实例
    """
    _model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    if use_api:
        if not _OPENAI_AVAILABLE:
            raise ImportError("请安装 langchain-openai: pip install langchain-openai")

        return OpenAIEmbeddings(
            model=_model_name,
            openai_api_base=os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")),
            openai_api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "ollama"))
        )
    else:
        if not _HF_AVAILABLE:
            raise ImportError("请安装 langchain-huggingface: pip install langchain-huggingface")

        return HuggingFaceEmbeddings(
            model_name=_model_name,
            model_kwargs={
                "device": "gpu",
                "backend": "onnx"
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 16
            }
        )