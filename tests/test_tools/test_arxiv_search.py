# tests/test_tools/test_arxiv_search.py
"""
测试 arXiv 搜索工具
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from tools.arxiv_search import search_arxiv


class TestArxivSearchTool:
    """测试 arXiv 搜索工具"""

    @pytest.fixture
    def mock_arxiv_result(self):
        """模拟 arXiv 搜索结果"""
        mock = MagicMock()
        mock.title = "Attention Is All You Need"

        # ✅ 关键修复：源码使用 a.name 访问作者名
        # 必须显式设置 .name 属性为字符串，而非依赖 __str__
        author1 = MagicMock()
        author1.name = "Ashish Vaswani"
        author2 = MagicMock()
        author2.name = "Noam Shazeer"
        mock.authors = [author1, author2]

        mock.published = datetime(2017, 6, 12)
        mock.summary = "The dominant sequence transduction models are based on complex recurrent neural networks."
        mock.pdf_url = "https://arxiv.org/pdf/1706.03762"
        return mock

    @patch('tools.arxiv_search.arxiv.Client')
    def test_successful_search(self, mock_client, mock_arxiv_result):
        """测试成功搜索"""
        mock_client_instance = MagicMock()
        mock_client_instance.results.return_value = [mock_arxiv_result]
        mock_client.return_value = mock_client_instance

        result = search_arxiv.invoke({
            "query": "attention mechanism",
            "max_results": 3
        })

        assert "Attention Is All You Need" in result
        assert "arxiv.org" in result or "PDF" in result

    @patch('tools.arxiv_search.arxiv.Client')
    def test_empty_results(self, mock_client):
        """测试空结果"""
        mock_client_instance = MagicMock()
        mock_client_instance.results.return_value = []
        mock_client.return_value = mock_client_instance

        result = search_arxiv.invoke({
            "query": "nonexistent_paper_xyz",
            "max_results": 3
        })

        assert "未找到" in result or "no" in result.lower()

    @patch('tools.arxiv_search.arxiv.Client')
    def test_exception_handling(self, mock_client):
        """测试异常处理"""
        mock_client.side_effect = Exception("Connection Error")

        result = search_arxiv.invoke({
            "query": "test",
            "max_results": 3
        })

        assert "出错" in result or "error" in result.lower()