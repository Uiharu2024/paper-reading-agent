# tests/test_memory/test_vector_store.py
"""
测试向量存储模块 vector_store.py
"""

import pytest
from unittest.mock import patch, MagicMock
from memory.vector_store import VectorStoreManager


@pytest.fixture
def mock_embedding():
    """Mock embedding 模型"""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 768]
    mock.embed_query.return_value = [0.1] * 768
    return mock


# ✅ 补回被误删的 vector_manager fixture
@pytest.fixture
def vector_manager(mock_embedding):
    """创建 VectorStoreManager 实例"""
    with patch('memory.vector_store.HuggingFaceEmbeddings', return_value=mock_embedding):
        manager = VectorStoreManager()
        yield manager


@pytest.fixture
def annotation_kwargs():
    """add_annotation 的标准参数"""
    return {
        "term": "self-attention",
        "explanation": "自注意力机制用于计算序列中每个位置与其他所有位置的相关性",
        "context": "The self-attention mechanism allows...",
        "paper_id": "paper_001",
        "paper_title": "Attention Is All You Need",
        "term_type": "CONCEPT",
        "term_analysis": {"domain_category": "NLP"},
        "page": 3,
    }


class TestVectorStoreManager:
    def test_add_annotation(self, vector_manager, annotation_kwargs):
        """测试添加划词记录"""
        result = vector_manager.add_annotation(**annotation_kwargs)
        assert isinstance(result, str)  # 返回值为 annotation_id

    def test_search_similar(self, vector_manager, annotation_kwargs):
        """测试相似性搜索"""
        vector_manager.add_annotation(**annotation_kwargs)
        # ✅ 使用正确的参数名 k
        results = vector_manager.search_similar("attention", k=3)
        assert isinstance(results, list)

    def test_get_paper_history(self, vector_manager, annotation_kwargs):
        """✅ 修正方法名：get_annotation_history → get_paper_history"""
        vector_manager.add_annotation(**annotation_kwargs)
        history = vector_manager.get_paper_history("paper_001")
        assert isinstance(history, list)
        assert len(history) >= 1

    def test_delete_annotation(self, vector_manager, annotation_kwargs):
        """测试删除标注"""
        ann_id = vector_manager.add_annotation(**annotation_kwargs)

        # ✅ 使用正确的参数名 doc_id，并检查返回值
        deleted = vector_manager.delete_annotation(doc_id=ann_id)
        assert deleted is True

        history = vector_manager.get_paper_history("paper_001")
        remaining_ids = [item.get("id") for item in history]
        assert ann_id not in remaining_ids, \
            f"Annotation {ann_id} was not deleted. Remaining IDs: {remaining_ids}"