# models/llm.py
"""
LLM 客户端统一管理

"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from functools import lru_cache
from langchain_openai import ChatOpenAI


class LLMTaskType(str, Enum):
    """定义系统中的标准任务类型"""
    ROUTER = "router"
    RECOGNIZER = "recognizer"
    RETRIEVER = "retriever"
    EXPLAINER = "explainer"
    EXPLAINER_FAST = "explainer_fast"
    REPORTER = "reporter"



_MODEL_REGISTRY: Dict[LLMTaskType, Dict[str, Any]] = {
    LLMTaskType.ROUTER: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.1,
        "max_tokens": 256,       
        "num_ctx": 2048,         
    },
    LLMTaskType.RECOGNIZER: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.2,
        "max_tokens": 512,
        "num_ctx": 4096,
    },
    LLMTaskType.RETRIEVER: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.1,
        "max_tokens": 1024,
        "num_ctx": 4096,
    },
    LLMTaskType.EXPLAINER_FAST: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.3,
        "max_tokens": 1024,
        "num_ctx": 4096,
    },
    LLMTaskType.EXPLAINER: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.4,
        "max_tokens": 4096,      
        "num_ctx": 8192,         
    },
    LLMTaskType.REPORTER: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.3,
        "max_tokens": 4096,      
        "num_ctx": 8192,
    }
}


@lru_cache(maxsize=1)  # ⚠️ 关键: 改为 maxsize=1，确保全局只有一个模型实例
def get_llm(
        task_type: LLMTaskType | str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
) -> ChatOpenAI:

    if isinstance(task_type, str):
        task_type = LLMTaskType(task_type)

    config = _MODEL_REGISTRY.get(task_type)
    if not config:
        raise ValueError(f"未知的任务类型: {task_type}")

    _base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    _api_key = api_key or os.getenv("LLM_API_KEY", "ollama")

    extra_body = {}
    if "num_ctx" in config:
        extra_body["num_ctx"] = config["num_ctx"]

    llm = ChatOpenAI(
        model=config["model_name"],
        base_url=_base_url,
        api_key=_api_key,
        temperature=config.get("temperature", 0.2),
        max_tokens=config.get("max_tokens", 2048),
        extra_body=extra_body if extra_body else None,
        streaming=True,
        request_timeout=120,
    )

    return llm
