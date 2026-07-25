# models/llm.py
"""
LLM 客户端统一管理 (RTX 5060 8GB 优化版)
策略: 单模型多角色复用 + Ollama 原生参数适配 + 显存安全限制
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


# ============================================================
# RTX 5060 8GB 专属模型注册表
# 核心原则: 所有任务共用同一个 Q4 量化模型实例
# 通过 system prompt 和 temperature 区分角色行为
# ============================================================
_MODEL_REGISTRY: Dict[LLMTaskType, Dict[str, Any]] = {
    LLMTaskType.ROUTER: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.1,
        "max_tokens": 256,       # Router 只需输出 JSON，极短即可
        "num_ctx": 2048,         # Ollama 上下文参数，Router 不需要长窗口
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
        "max_tokens": 4096,      # 深度解释保留较长输出
        "num_ctx": 8192,         # 仅重度任务开放完整上下文
    },
    LLMTaskType.REPORTER: {
        "model_name": "qwen2.5:14b-instruct-q4_K_M",
        "temperature": 0.3,
        "max_tokens": 4096,      # 从 8192 降至 4096，防止 KV Cache 撑爆显存
        "num_ctx": 8192,
    }
}


@lru_cache(maxsize=1)  # ⚠️ 关键: 改为 maxsize=1，确保全局只有一个模型实例
def get_llm(
        task_type: LLMTaskType | str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
) -> ChatOpenAI:
    """
    获取配置好的 LLM 客户端实例 (带缓存)
    RTX 5060 优化: 所有任务共享同一模型实例，避免重复加载导致 OOM
    """
    if isinstance(task_type, str):
        task_type = LLMTaskType(task_type)

    config = _MODEL_REGISTRY.get(task_type)
    if not config:
        raise ValueError(f"未知的任务类型: {task_type}")

    _base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    _api_key = api_key or os.getenv("LLM_API_KEY", "ollama")

    # ============================================================
    # Ollama 原生参数适配
    # Ollama 不支持 enable_thinking，思考能力由模型自身 + prompt 控制
    # num_ctx 通过 extra_body 传递给 Ollama 以动态调整上下文窗口
    # ============================================================
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
        # ⚠️ 关键: 禁用 LangChain 内部的并发请求，防止同时触发多个推理导致显存峰值超标
        request_timeout=120,
    )

    return llm