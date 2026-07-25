from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from config.settings import inference_config


def get_embeddings():
    """统一 Embedding 工厂函数"""
    cfg = inference_config

    if cfg.backend == "cloud":
        if not cfg.cloud_api_key:
            raise ValueError("云端模式已启用，但未设置 CLOUD_API_KEY 环境变量")
        return OpenAIEmbeddings(
            model=cfg.cloud_embedding_model,
            base_url=cfg.cloud_embedding_base_url,
            api_key=cfg.cloud_api_key,
        )
    else:  # 默认 local
        return OllamaEmbeddings(
            model=cfg.local_embedding_model,
            base_url=cfg.local_base_url.replace("/v1", "")
        )