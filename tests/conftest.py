# tests/conftest.py
"""
pytest 全局配置文件
定义共享的 fixtures 和测试环境配置
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# 设置测试环境变量
os.environ["LLM_BASE_URL"] = "http://localhost:8000/v1"
os.environ["LLM_API_KEY"] = "test-key"
os.environ["EMBEDDING_MODEL"] = "test-model"
os.environ["USE_GPU"] = "false"


@pytest.fixture
def sample_paper_text():
    """示例论文文本"""
    return """
    Attention Is All You Need

    Abstract: The dominant sequence transduction models are based on complex 
    recurrent or convolutional neural networks that include an encoder and a decoder.
    The best performing models also connect the encoder and decoder through an 
    attention mechanism. We propose a new simple network architecture, the 
    Transformer, based solely on attention mechanisms.

    1. Introduction
    Recurrent neural networks, long short-term memory and gated recurrent neural 
    networks in particular, have been firmly established as state of the art 
    approaches in sequence modeling and transduction problems.
    """


@pytest.fixture
def sample_annotation():
    """示例划词记录"""
    return {
        "term": "Self-Attention",
        "term_type": "CONCEPT",
        "standard_name_zh": "自注意力机制",
        "standard_name_en": "Self-Attention",
        "domain_category": "自然语言处理",
        "explanation_summary": "自注意力机制是一种注意力机制，用于计算序列中每个位置与其他所有位置的相关性。",
        "paper_id": "paper_001",
        "paper_title": "Attention Is All You Need",
        "page": 3,
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def sample_graph_data():
    """示例知识图谱数据"""
    return {
        "nodes": [
            {"id": "self-attention", "label": "Self-Attention", "type": "CONCEPT", "query_count": 3},
            {"id": "transformer", "label": "Transformer", "type": "METHOD", "query_count": 2},
            {"id": "encoder", "label": "Encoder", "type": "CONCEPT", "query_count": 1},
        ],
        "edges": [
            {"source": "self-attention", "target": "transformer", "type": "part_of", "weight": 1.0},
            {"source": "transformer", "target": "encoder", "type": "has_component", "weight": 1.0},
        ]
    }


@pytest.fixture
def mock_llm():
    """模拟 LLM 客户端"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content='{"term_type": "CONCEPT", "confidence": 0.95}')
    return mock


@pytest.fixture
def mock_embedding():
    """模拟 Embedding 模型"""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock