# tests/test_tools/test_web_search.py
"""
测试 DuckDuckGo 网络搜索工具
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.web_search import search_web


class TestWebSearchTool:
    """测试网络搜索工具"""

    @pytest.fixture
    def mock_ddg_results(self):
        """模拟 DuckDuckGo 搜索结果"""
        return [
            {
                "title": "Self-Attention - Wikipedia",
                "href": "https://en.wikipedia.org/wiki/Self-attention",
                "body": "Self-attention is a mechanism used in transformer models."
            },
            {
                "title": "Understanding Attention Mechanisms",
                "href": "https://example.com/attention",
                "body": "Attention mechanisms allow models to focus on relevant parts."
            }
        ]

    @patch('tools.web_search.DDGS')
    def test_successful_search(self, mock_ddgs, mock_ddg_results):
        """测试成功搜索"""
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_ddg_results
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs.return_value = mock_instance

        result = search_web.invoke({
            "query": "Self-Attention",
            "max_results": 3
        })

        assert "Self-Attention" in result
        assert "Wikipedia" in result or "wikipedia" in result

    @patch('tools.web_search.DDGS')
    def test_empty_results(self, mock_ddgs):
        """测试空结果"""
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs.return_value = mock_instance

        result = search_web.invoke({
            "query": "nonexistent_query_xyz",
            "max_results": 3
        })

        assert "未找到" in result or "no" in result.lower()

    @patch('tools.web_search.DDGS')
    def test_exception_handling(self, mock_ddgs):
        """测试异常处理"""
        mock_ddgs.side_effect = Exception("Rate Limit Exceeded")

        result = search_web.invoke({
            "query": "test",
            "max_results": 3
        })

        assert "出错" in result or "error" in result.lower() or "频率" in result