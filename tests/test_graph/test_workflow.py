# tests/test_graph/test_workflow.py
"""
测试 LangGraph 工作流
"""

import pytest
from unittest.mock import patch, MagicMock
from graph.workflow import get_graph
from graph.state import PaperReadingState


class TestWorkflow:
    """测试 LangGraph 工作流"""

    @pytest.fixture
    def graph(self):
        """获取编译后的 Graph"""
        return get_graph()

    def test_graph_compilation(self, graph):
        """测试 Graph 是否成功编译"""
        assert graph is not None
        assert hasattr(graph, 'invoke')
        assert hasattr(graph, 'stream')

    def test_graph_execution(self):
        """测试图执行"""
        from graph.workflow import build_graph
        graph = build_graph(use_memory_saver=True)
        assert graph is not None
        from graph.workflow import get_graph
        compiled = get_graph()
        assert compiled is not None
        assert hasattr(compiled, 'invoke')
        assert hasattr(compiled, 'stream')

    def test_graph_state_structure(self):
        """测试状态结构"""
        # ✅ 修复：PaperReadingState 是 TypedDict/dict，用 [] 访问而非 .
        from graph.state import PaperReadingState
        state: PaperReadingState = {
            "selected_text": "attention",
            "selection_context": "context",
            "paper_domain": "NLP",
            "annotation_history": [],
            "messages": [],
        }
        assert state["selected_text"] == "attention"
        assert isinstance(state, dict)