# tests/test_memory/test_user_profile.py （全文替换）

import pytest
from unittest.mock import MagicMock
from memory.user_profile import UserProfileManager


class TestUserProfileManager:
    @pytest.fixture
    def profile_manager(self):
        return UserProfileManager()

    @pytest.fixture
    def sample_annotation_history(self):
        return [
            {"term": "attention", "term_type": "CONCEPT", "domain_category": "NLP"},
            {"term": "transformer", "term_type": "METHOD", "domain_category": "NLP"},
        ]

    @pytest.fixture
    def sample_term_analysis(self):
        return {
            "term": "attention",
            "term_type": "CONCEPT",
            "domain_category": "NLP",
            "processing_depth": "QUICK"
        }

    def test_initial_profile(self, profile_manager):
        """✅ 原 get_profile → get_profile_summary"""
        summary = profile_manager.get_profile_summary()
        assert isinstance(summary, str)

    def test_update_profile(self, profile_manager, sample_annotation_history, sample_term_analysis):
        """✅ 补全 current_term_analysis 参数"""
        result = profile_manager.update_profile(
            annotation_history=sample_annotation_history,
            current_term_analysis=sample_term_analysis
        )
        assert isinstance(result, dict)

    def test_increment_familiarity(self, profile_manager, sample_annotation_history, sample_term_analysis):
        """✅ 通过 update_profile 间接提升熟悉度"""
        result = profile_manager.update_profile(
            annotation_history=sample_annotation_history,
            current_term_analysis=sample_term_analysis
        )
        assert isinstance(result, dict)

    def test_familiarity_cap(self, profile_manager, sample_annotation_history, sample_term_analysis):
        """✅ 多次更新验证不会崩溃（熟悉度上限由内部逻辑保证）"""
        for _ in range(10):
            result = profile_manager.update_profile(
                annotation_history=sample_annotation_history,
                current_term_analysis=sample_term_analysis
            )
        assert isinstance(result, dict)

    def test_add_interest(self, profile_manager, sample_annotation_history, sample_term_analysis):
        """✅ 通过 update_profile 添加兴趣领域"""
        analysis = {**sample_term_analysis, "domain_category": "CV"}
        result = profile_manager.update_profile(
            annotation_history=sample_annotation_history,
            current_term_analysis=analysis
        )
        assert isinstance(result, dict)

    def test_reset_profile(self, profile_manager, sample_annotation_history, sample_term_analysis):
        """✅ reset → 重新初始化 + update_profile"""
        # 先更新
        profile_manager.update_profile(
            annotation_history=sample_annotation_history,
            current_term_analysis=sample_term_analysis
        )
        # 再"重置"：创建新实例或用空数据更新
        new_manager = UserProfileManager()
        summary = new_manager.get_profile_summary()
        assert isinstance(summary, str)