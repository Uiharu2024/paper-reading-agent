import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InferenceConfig:
    """推理后端配置，优先读取环境变量，否则使用默认值"""
    backend: Literal["local", "cloud"] = os.getenv("INFERENCE_BACKEND", "local")

    # 本地配置 (Ollama)
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434/v1")
    local_model: str = os.getenv("LOCAL_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "bge-m3")

    # 云端配置 (Qwen / DashScope)
    cloud_base_url: str = os.getenv("CLOUD_BASE_URL", "")
    cloud_model: str = os.getenv("CLOUD_MODEL", "")
    cloud_api_key: str = os.getenv("CLOUD_API_KEY", "")
    cloud_embedding_model: str = os.getenv("CLOUD_EMBEDDING_MODEL", "")
    cloud_embedding_base_url: str = os.getenv("CLOUD_EMBEDDING_BASE_URL","")


# 全局单例，避免重复读取
inference_config = InferenceConfig()