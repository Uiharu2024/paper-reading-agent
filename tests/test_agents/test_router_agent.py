# tests/test_agents/test_router_agent.py
import pytest
from unittest.mock import patch, MagicMock
from agents.router_agent import router_node


class TestRouterAgent:
    @pytest.fixture
    def make_state(self):
        def _make(**overrides):
            base = {
                "selected_text": "attention mechanism",
                "selection_context": "This paper proposes...",
                "paper_domain": "NLP",
                "annotation_history": [],
            }
            base.update(overrides)
            return base
        return _make

    @patch('agents.router_agent.router_prompt')
    def test_route_simple_query(self, mock_prompt, make_state):
        """测试简单查询路由"""
        # ✅ 核心修复：patch router_prompt 的 __or__，拦截 LCEL 链构建
        mock_chain = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        mock_decision = MagicMock()
        mock_decision.term_type = "CONCEPT"
        mock_decision.processing_depth = "QUICK"
        mock_decision.brief_reasoning = "simple query"
        mock_chain.invoke.return_value = mock_decision

        state = make_state()
        result = router_node(state)

        assert result["processing_depth"] == "QUICK"
        assert result["term_type"] == "CONCEPT"
        assert result["current_agent"] == "router"

    @patch('agents.router_agent.router_prompt')
    def test_route_complex_query(self, mock_prompt, make_state):
        """测试复杂查询路由"""
        mock_chain = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        mock_decision = MagicMock()
        mock_decision.term_type = "METHOD"
        mock_decision.processing_depth = "DEEP"
        mock_decision.brief_reasoning = "complex analysis"
        mock_chain.invoke.return_value = mock_decision

        state = make_state(selected_text="transformer architecture with multi-head attention")
        result = router_node(state)

        assert result["processing_depth"] == "DEEP"

    @patch('agents.router_agent.router_prompt')
    def test_route_with_explicit_depth(self, mock_prompt, make_state):
        """测试路由节点正常返回结构"""
        mock_chain = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        mock_decision = MagicMock()
        mock_decision.term_type = "CONCEPT"
        mock_decision.processing_depth = "DEEP"
        mock_decision.brief_reasoning = "explicit"
        mock_chain.invoke.return_value = mock_decision

        state = make_state()
        result = router_node(state)

        assert isinstance(result, dict)
        assert "processing_depth" in result
        assert "messages" in result