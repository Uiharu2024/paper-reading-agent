# tests/test_tools/test_scholar_api.py
"""
测试 Semantic Scholar API 工具
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.scholar_api import search_semantic_scholar


class TestSemanticScholarTool:
    """测试 Semantic Scholar 搜索工具"""

    @pytest.fixture
    def mock_response(self):
        """模拟 API 响应"""
        return {
            "data": [
                {
                    "title": "Attention Is All You Need",
                    "year": 2017,
                    "venue": "NeurIPS",
                    "citationCount": 80000,
                    "authors": [
                        {"name": "Ashish Vaswani"},
                        {"name": "Noam Shazeer"}
                    ],
                    "abstract": "The dominant sequence transduction models..."
                }
            ]
        }

    @patch('tools.scholar_api.requests.get')
    def test_successful_search(self, mock_get, mock_response):
        """测试成功搜索"""
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_semantic_scholar.invoke({
            "term": "Self-Attention",
            "limit": 3
        })

        assert "Attention Is All You Need" in result
        assert "80000" in result or "citation" in result.lower()

    @patch('tools.scholar_api.requests.get')
    def test_empty_results(self, mock_get):
        """测试空结果"""
        mock_get.return_value.json.return_value = {"data": []}
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_semantic_scholar.invoke({
            "term": "nonexistent_term_xyz",
            "limit": 3
        })

        assert "未找到" in result or "no" in result.lower()

    @patch('tools.scholar_api.requests.get')
    def test_api_error_handling(self, mock_get):
        """测试 API 错误处理"""
        # ✅ 修复：让 requests.get 返回一个带 raise_for_status 抛异常的响应
        # 而不是直接 side_effect=Exception（那会在 requests.get 调用时就抛出）
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Network Error")
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = search_semantic_scholar.invoke({
            "term": "test",
            "limit": 3
        })

        assert "失败" in result or "error" in result.lower() or "出错" in result

    @patch('tools.scholar_api.requests.get')
    def test_rate_limit_handling(self, mock_get):
        """测试 Rate Limit 处理"""
        # 第一次返回 429，第二次成功
        mock_get.side_effect = [
            MagicMock(status_code=429),
            MagicMock(
                status_code=200,
                json=lambda: {"data": []},
                raise_for_status=MagicMock()
            )
        ]

        result = search_semantic_scholar.invoke({
            "term": "test",
            "limit": 1
        })

        # 应该成功重试
        assert result is not None