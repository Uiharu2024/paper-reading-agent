# app.py
"""
论文阅读多智能体系统 - Streamlit 交互主界面

运行方式:
    streamlit run app.py --server.port 8501
"""

import os
import asyncio
import streamlit as st
from datetime import datetime
from uuid import uuid4

# 导入项目核心模块
from graph.workflow import get_graph
from graph.state import PaperReadingState
from memory.memory_manager import MemoryManager
from models.schemas import ReadingRequest
from config.settings import InferenceConfig
import config.settings as settings_module

# ==================== 1. 页面基础配置 ====================
st.set_page_config(
    page_title="AI 论文阅读助手 (Qwen3-Multi-Agent)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 2. 会话状态初始化 ====================
if "graph_app" not in st.session_state:
    st.session_state.graph_app = get_graph()
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager()
if "history_logs" not in st.session_state:
    st.session_state.history_logs = []

# ==================== 3. 侧边栏 (Sidebar) ====================
with st.sidebar:
    st.title("⚙️ 系统控制台")

    # ---------- 3.0 推理后端配置  ----------
    st.header("🔌 推理后端")
    backend = st.radio(
        "选择模式",
        ["local", "cloud"],
        index=0 if settings_module.inference_config.backend == "local" else 1,
        format_func=lambda x: "🖥️ 本地 Ollama" if x == "local" else "☁️ 云端 Qwen",
        horizontal=True
    )

    if backend == "local":
        local_model = st.text_input("LLM 模型", value=settings_module.inference_config.local_model)
        local_base_url = st.text_input("Ollama URL", value=settings_module.inference_config.local_base_url)
        local_emb = st.text_input("Embedding 模型", value=settings_module.inference_config.local_embedding_model)
    else:
        cloud_model = st.selectbox(
            "LLM 模型",
            ["qwen-plus", "qwen-max", "qwen-turbo"],
            index=["qwen-plus", "qwen-max", "qwen-turbo"].index(settings_module.inference_config.cloud_model)
            if settings_module.inference_config.cloud_model in ["qwen-plus", "qwen-max", "qwen-turbo"] else 0
        )
        cloud_key = st.text_input("API Key", type="password", value=settings_module.inference_config.cloud_api_key)
        cloud_url = st.text_input("Base URL", value=settings_module.inference_config.cloud_base_url)
        cloud_emb = st.text_input("Embedding 模型", value=settings_module.inference_config.cloud_embedding_model)

    if st.button("✅ 应用模型配置", use_container_width=True):
        os.environ["INFERENCE_BACKEND"] = backend
        if backend == "local":
            os.environ.update({
                "LOCAL_MODEL": local_model,
                "LOCAL_BASE_URL": local_base_url,
                "LOCAL_EMBEDDING_MODEL": local_emb
            })
        else:
            os.environ.update({
                "CLOUD_MODEL": cloud_model,
                "CLOUD_API_KEY": cloud_key,
                "CLOUD_BASE_URL": cloud_url,
                "CLOUD_EMBEDDING_MODEL": cloud_emb
            })
        # 刷新配置单例 + 重建 Graph/Memory
        settings_module.inference_config = InferenceConfig()
        st.session_state.graph_app = get_graph()
        st.session_state.memory_manager = MemoryManager()
        st.success(f"已切换到 {'本地' if backend == 'local' else '云端'} | LLM: {settings_module.inference_config.cloud_model if backend == 'cloud' else settings_module.inference_config.local_model}")
        st.rerun()

    st.caption(f"当前: **{settings_module.inference_config.backend}** | "
               f"{settings_module.inference_config.cloud_model if settings_module.inference_config.backend == 'cloud' else settings_module.inference_config.local_model}")
    st.divider()

    # ---------- 3.1 论文上下文配置  ----------
    st.header("📄 当前论文")
    paper_title = st.text_input("论文标题", value="Attention Is All You Need")
    paper_domain = st.selectbox("学科领域", ["自然语言处理", "计算机视觉", "机器学习", "通用学术"], index=0)
    paper_id = st.text_input("论文 ID (用于记忆隔离)", value="paper_001")

    # ---------- 3.2 用户画像展示  ----------
    st.header("🧠 用户画像 (动态推断)")
    profile = st.session_state.memory_manager.user_profile.profile
    st.metric("学科背景", profile.get("background", "unknown"))
    st.metric("熟悉度", "⭐" * profile.get("familiarity", 1))
    st.metric("偏好深度", profile.get("preferred_depth", "STANDARD"))

    # ---------- 3.3 记忆统计  ----------
    st.header("📊 记忆统计")
    stats = st.session_state.memory_manager.get_user_statistics()
    st.metric("总划词数", stats.get("total_annotations", 0))
    st.metric("知识图谱节点", len(st.session_state.memory_manager.knowledge_graph.graph.nodes))

    if st.button("🗑️ 清空当前会话记忆"):
        st.session_state.history_logs = []
        st.session_state.memory_manager = MemoryManager()
        st.rerun()

# ==================== 4. 主界面 (Main Area) ====================
st.title("📖 AI 论文阅读助手")
st.markdown("基于 **LangGraph + Qwen3 混合推理** 的多智能体系统。划选文本，AI 将自动识别、检索并生成深度解释。")

# 4.1 划词输入区
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("✍️ 划词输入")
    selected_text = st.text_input("选中的术语/句子", placeholder="例如：Self-Attention")

    processing_depth = st.radio(
        "处理深度 (可覆盖自动路由)",
        ["AUTO (由 Router 决定)", "QUICK (快速)", "STANDARD (标准)", "DEEP (深度)"],
        index=0
    )
    depth_value = "" if "AUTO" in processing_depth else processing_depth.split(" ")[0]

with col2:
    st.subheader("📝 上下文段落")
    selection_context = st.text_area(
        "术语所在的段落上下文",
        height=150,
        value="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
    )

# 4.2 执行按钮与核心逻辑
if st.button("🚀 开始智能分析", type="primary", use_container_width=True):
    if not selected_text:
        st.warning("⚠️ 请输入划选的术语或句子！")
    else:
        # 构建初始 State
        thread_id = f"thread_{uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "paper_domain": paper_domain,
            "selected_text": selected_text,
            "selection_context": selection_context,
            "selection_page": 1,
            "processing_depth": depth_value if depth_value else "STANDARD",
            "annotation_history": st.session_state.history_logs,
            "user_interest_profile": profile,
            "refinement_count": 0,
            "needs_refinement": False
        }

        # 使用 st.status 展示多智能体执行过程
        with st.status(f"🤖 正在分析「{selected_text}」...", expanded=True) as status:
            st.write("📡 **Router Agent**: 正在分析意图与处理深度...")


            # 异步执行 LangGraph
            async def _collect_graph_output():
                node_outputs = {}
                async for event in st.session_state.graph_app.astream(
                        initial_state, config=config, stream_mode="updates"
                ):
                    for node_name, updates in event.items():
                        node_outputs[node_name] = updates

                        if node_name == "router":
                            st.write("🔍 **Recognizer Agent**: 正在识别术语类型...")
                        elif node_name == "recognizer":
                            st.write("🌐 **Retriever Agent**: 正在检索...")
                        elif node_name == "retriever":
                            st.write("🧠 **Explainer Agent**: 正在生成解释...")
                        elif node_name == "explainer":
                            st.write("📊 **Reporter Agent**: 正在生成知识卡片...")
                        elif node_name == "reporter":
                            st.write("💾 **Memory Updater**: 正在更新记忆...")
                return node_outputs


            try:
                node_outputs = asyncio.run(_collect_graph_output())
                status.update(label="✅ 分析完成！", state="complete", expanded=False)
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                status.update(label=f"❌ 执行出错: {type(e).__name__}", state="error", expanded=True)
                st.error(error_msg)
                st.stop()

        # ==================== 5. 结果渲染区 ====================
        st.divider()

        final_explanation = ""
        final_report = ""
        term_analysis = {}

        # 遍历节点输出寻找关键字段
        for node, upd in node_outputs.items():
            if "explanation" in upd: final_explanation = upd["explanation"]
            if "session_report" in upd: final_report = upd["session_report"]
            if "term_analysis" in upd: term_analysis = upd["term_analysis"]

        # 如果流式没抓到，尝试从 memory_manager 获取最新状态
        if not final_explanation:
            final_explanation = "解释生成中... (若未显示请检查 LLM 服务)"

        # 5.1 术语分析结果 (Recognizer 输出)
        if term_analysis:
            with st.expander("🔍 术语结构化分析 (Recognizer)", expanded=False):
                st.json(term_analysis)

        # 5.2 详细解释 (Explainer 输出)
        st.subheader("💡 深度解释")
        st.markdown(final_explanation if final_explanation else "暂无解释内容")

        # 5.3 知识卡片 (Reporter 输出)
        if final_report:
            with st.expander("📇 即时知识卡片 (Reporter)", expanded=True):
                st.markdown(final_report)

        # 5.4 知识图谱可视化 (Memory 输出)
        st.subheader("🕸️ 局部知识图谱")
        kg_data = st.session_state.memory_manager.get_paper_knowledge_graph(paper_id)
        if kg_data and kg_data.get("nodes"):
            st.dataframe(kg_data["nodes"])
            st.dataframe(kg_data["edges"])
        else:
            st.info("图谱数据积累中，多查几个词即可看到关联网络。")

        # 记录到历史
        st.session_state.history_logs.append({
            "term": selected_text,
            "term_type": term_analysis.get("term_type", "UNKNOWN"),
            "timestamp": datetime.now().isoformat()
        })
