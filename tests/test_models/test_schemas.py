# tests/test_models/test_schemas.py
"""
测试数据模型 schemas.py
"""

import pytest
from datetime import datetime
from models.schemas import (
    ReadingRequest,
    ReadingResponse,
    AnnotationRecord,
    KnowledgeEdge,
    UserProfileData
)


class TestReadingRequest:
    """测试 ReadingRequest 模型"""

    def test_valid_request(self):
        """测试有效的请求"""
        req = ReadingRequest(
            paper_id="paper_001",
            paper_title="Test Paper",
            selected_text="Self-Attention",
            selection_context="This is the context."
        )
        assert req.paper_id == "paper_001"
        assert req.selected_text == "Self-Attention"
        assert req.force_deep_analysis == False

    def test_missing_required_fields(self):
        """测试缺少必填字段"""
        with pytest.raises(Exception):
            ReadingRequest(paper_id="paper_001")


class TestAnnotationRecord:
    """测试 AnnotationRecord 模型"""

    def test_valid_annotation(self, sample_annotation):
        """测试有效的划词记录"""
        record = AnnotationRecord(**sample_annotation)
        assert record.term == "Self-Attention"
        assert record.term_type == "CONCEPT"
        assert record.standard_name_zh == "自注意力机制"

    def test_timestamp_auto_generated(self):
        """测试时间戳自动生成"""
        record = AnnotationRecord(
            term="Test",
            term_type="CONCEPT",
            paper_id="paper_001"
        )
        assert record.timestamp is not None
        # 验证是 ISO 格式
        datetime.fromisoformat(record.timestamp)


class TestKnowledgeEdge:
    """测试 KnowledgeEdge 模型"""

    def test_valid_edge(self):
        """测试有效的知识图谱边"""
        edge = KnowledgeEdge(
            source="self-attention",
            target="transformer",
            relation_type="part_of",
            weight=1.0,
            paper_ids=["paper_001"]
        )
        assert edge.source == "self-attention"
        assert edge.target == "transformer"
        assert edge.weight == 1.0

    def test_default_values(self):
        """测试默认值"""
        edge = KnowledgeEdge(
            source="a",
            target="b"
        )
        assert edge.relation_type == "related"
        assert edge.weight == 1.0
        assert edge.paper_ids == []


class TestUserProfileData:
    """测试 UserProfileData 模型"""

    def test_valid_profile(self):
        """测试有效的用户画像"""
        profile = UserProfileData(
            background="自然语言处理",
            familiarity=3,
            interests=["Transformer", "Attention"],
            preferred_depth="DEEP"
        )
        assert profile.background == "自然语言处理"
        assert profile.familiarity == 3
        assert len(profile.interests) == 2

    def test_familiarity_bounds(self):
        """测试熟悉度边界值"""
        # 最小值
        profile = UserProfileData(familiarity=1)
        assert profile.familiarity == 1

        # 最大值
        profile = UserProfileData(familiarity=5)
        assert profile.familiarity == 5

        # 超出范围
        with pytest.raises(Exception):
            UserProfileData(familiarity=0)

        with pytest.raises(Exception):
            UserProfileData(familiarity=6)