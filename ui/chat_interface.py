# ui/chat_interface.py
"""
对话式交互界面组件

功能:
1. 展示多智能体的思考过程 (白盒化)
2. 支持用户反馈和追问
3. 历史对话记录管理
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatInterface:
    """对话式交互界面"""

    def __init__(self):
        """初始化对话界面"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "current_status" not in st.session_state:
            st.session_state.current_status = None

    def render(self):
        """渲染完整的对话界面"""
        st.subheader("💬 AI 助手对话")

        # 1. 显示历史对话
        self._render_history()

        # 2. 显示当前处理状态
        self._render_current_status()

        # 3. 用户输入区
        self._render_input_area()

    def _render_history(self):
        """渲染历史对话记录"""
        history = st.session_state.chat_history

        if not history:
            st.info("👋 你好！我是你的 AI 论文阅读助手。划选论文中的文本，我会为你智能分析。")
            return

        # 使用容器展示对话历史
        chat_container = st.container()
        with chat_container:
            for msg in history:
                self._render_message(msg)

    def _render_message(self, msg: Dict[str, Any]):
        """渲染单条消息"""
        role = msg.get("role", "user")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        agent_info = msg.get("agent_info", {})

        if role == "user":
            # 用户消息
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**划选文本**: {msg.get('selected_text', '')}")
                if msg.get("context"):
                    with st.expander("📝 上下文", expanded=False):
                        st.text(msg.get("context", ""))
        else:
            # AI 消息
            with st.chat_message("assistant", avatar="🤖"):
                # 显示处理过程 (白盒化)
                if agent_info:
                    self._render_agent_process(agent_info)

                # 显示最终解释
                st.markdown(content)

                # 显示相关推荐
                if msg.get("related_terms"):
                    st.markdown("**🔗 相关概念**: " + ", ".join(msg.get("related_terms", [])))

                # 用户反馈按钮
                col1, col2, col3 = st.columns([1, 1, 6])
                with col1:
                    if st.button("👍 有帮助", key=f"up_{timestamp}"):
                        self._record_feedback(timestamp, "positive")
                with col2:
                    if st.button("👎 需改进", key=f"down_{timestamp}"):
                        self._record_feedback(timestamp, "negative")

    def _render_agent_process(self, agent_info: Dict[str, Any]):
        """渲染多智能体处理过程"""
        with st.expander("🔍 查看 AI 思考过程", expanded=False):
            st.markdown("#### 📡 Router Agent")
            st.text(f"意图识别: {agent_info.get('intent', 'N/A')}")
            st.text(f"处理深度: {agent_info.get('depth', 'N/A')}")

            st.markdown("#### 🔍 Recognizer Agent")
            term_analysis = agent_info.get("term_analysis", {})
            if term_analysis:
                st.json(term_analysis)

            st.markdown("#### 🌐 Retriever Agent")
            tools_used = agent_info.get("tools_used", [])
            if tools_used:
                for tool in tools_used:
                    st.text(f"✓ 调用工具: {tool}")

            st.markdown("#### 🧠 Explainer Agent")
            st.text(f"记忆注入: {agent_info.get('memory_injected', False)}")
            st.text(f"个性化等级: {agent_info.get('personalization_level', 'N/A')}")

    def _render_current_status(self):
        """渲染当前处理状态"""
        status = st.session_state.current_status
        if status:
            with st.status(f"🤖 {status.get('message', '处理中...')}", expanded=True) as s:
                steps = status.get("steps", [])
                for step in steps:
                    st.write(step)

    def _render_input_area(self):
        """渲染用户输入区"""
        st.divider()

        # 快速输入区
        col1, col2 = st.columns([3, 1])

        with col1:
            user_input = st.text_input(
                "快速划词输入",
                placeholder="输入你划选的术语或句子...",
                key="quick_input"
            )

        with col2:
            depth = st.selectbox(
                "分析深度",
                ["AUTO", "QUICK", "STANDARD", "DEEP"],
                index=0,
                key="depth_select"
            )

        if st.button("🚀 分析", type="primary", use_container_width=True):
            if user_input:
                return {
                    "selected_text": user_input,
                    "processing_depth": depth if depth != "AUTO" else "",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                st.warning("⚠️ 请输入划选的文本！")

        return None

    def add_message(self, role: str, content: str, **kwargs):
        """
        添加一条消息到历史记录

        Args:
            role: 消息角色 (user/assistant)
            content: 消息内容
            **kwargs: 其他元数据 (selected_text, context, agent_info等)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        st.session_state.chat_history.append(message)

    def update_status(self, message: str, steps: List[str] = None):
        """更新当前处理状态"""
        st.session_state.current_status = {
            "message": message,
            "steps": steps or []
        }

    def clear_status(self):
        """清除当前状态"""
        st.session_state.current_status = None

    def _record_feedback(self, timestamp: str, feedback_type: str):
        """记录用户反馈"""
        # 在实际应用中，这里应该将反馈发送到后端存储
        st.success(f"✅ 感谢反馈！已记录为 {feedback_type}")
        # TODO: 调用 memory_manager.update_feedback(timestamp, feedback_type)

    def get_latest_user_input(self) -> Optional[Dict[str, Any]]:
        """获取最新的用户输入 (用于触发 Graph 执行)"""
        return self._render_input_area()