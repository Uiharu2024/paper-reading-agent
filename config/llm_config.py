from langchain_openai import ChatOpenAI
from config.settings import inference_config


def get_llm(temperature=0.1, structured_output=None):
    """统一 LLM 工厂函数，根据 backend 自动切换本地/云端"""
    cfg = inference_config

    if cfg.backend == "cloud":
        if not cfg.cloud_api_key:
            raise ValueError("云端模式已启用，但未设置 CLOUD_API_KEY 环境变量")
        llm = ChatOpenAI(
            model=cfg.cloud_model,
            base_url=cfg.cloud_base_url,
            api_key=cfg.cloud_api_key,
            temperature=temperature,
        )
    else:  # 默认 local
        llm = ChatOpenAI(
            model=cfg.local_model,
            base_url=cfg.local_base_url,
            api_key="ollama",
            temperature=temperature,
        )

    if structured_output:
        return llm.with_structured_output(structured_output)
    return llm