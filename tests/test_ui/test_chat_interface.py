# tests/test_ui/test_chat_interface.py
"""
测试 ChatInterface UI 组件
"""

import pytest
from unittest.mock import patch, MagicMock
from ui.chat_interface import ChatInterface


class TestChatInterface:
    """测试 ChatInterface"""

    @pytest.fixture
    def chat_interface(self):
        """创建 ChatInterface 实例"""
        with patch('streamlit.session_state', MagicMock()):
            return ChatInterface()

    def test_add_message(self, chat_interface):
        """测试添加消息"""
        with patch('streamlit.session_state') as mock_state:
            mock_state.chat_history = []

            chat_interface.add_message(
                role="user",
                content="Test message",
                selected_text="Self-Attention"
            )

            assert len(mock_state.chat_history) == 1
            assert mock_state.chat_history[0]["role"] == "user"

    def test_update_status(self, chat_interface):
        """测试更新状态"""
        with patch('streamlit.session_state') as mock_state:
            mock_state.current_status = None

            chat_interface.update_status(
                message="Processing...",
                steps=["Step 1", "Step 2"]
            )

            assert mock_state.current_status is not None
            assert mock_state.current_status["message"] == "Processing..."

    def test_clear_status(self, chat_interface):
        """测试清除状态"""
        with patch('streamlit.session_state') as mock_state:
            mock_state.current_status = {"message": "test"}

            chat_interface.clear_status()

            assert mock_state.current_status is None