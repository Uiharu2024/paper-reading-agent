# ui/pdf_viewer.py
"""
PDF 文档阅读器组件

功能:
1. 渲染 PDF 文档页面
2. 支持文本划选交互 (模拟)
3. 显示划词高亮和注释
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO


class PDFViewer:
    """PDF 文档阅读器"""

    def __init__(self):
        """初始化 PDF 阅读器"""
        if "pdf_file" not in st.session_state:
            st.session_state.pdf_file = None
        if "pdf_pages" not in st.session_state:
            st.session_state.pdf_pages = []
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        if "annotations" not in st.session_state:
            st.session_state.annotations = []

    def render(self):
        """渲染 PDF 阅读器界面"""
        st.subheader("📄 论文阅读器")

        # 1. 文件上传区
        self._render_upload_area()

        # 2. PDF 显示区
        if st.session_state.pdf_file:
            self._render_pdf_display()

            # 3. 划词注释列表
            self._render_annotations_list()

    def _render_upload_area(self):
        """渲染文件上传区"""
        uploaded_file = st.file_uploader(
            "上传 PDF 论文",
            type=["pdf"],
            help="支持标准 PDF 格式，建议文件大小 < 50MB"
        )

        if uploaded_file:
            st.session_state.pdf_file = uploaded_file
            # 这里应该调用 PDF 解析服务提取文本和页面
            # 简化处理：直接存储文件
            st.success(f"✅ 已加载: {uploaded_file.name}")

    def _render_pdf_display(self):
        """渲染 PDF 显示区"""
        pdf_file = st.session_state.pdf_file

        if not pdf_file:
            return

        # 页面控制
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀️ 上一页"):
                st.session_state.current_page = max(1, st.session_state.current_page - 1)
        with col2:
            st.text(f"第 {st.session_state.current_page} 页")
        with col3:
            if st.button("下一页 ▶️"):
                st.session_state.current_page += 1

        # PDF 嵌入显示 (使用 iframe)
        # 注意: Streamlit 对 PDF 的原生支持有限，这里使用 base64 编码嵌入
        try:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            pdf_display = f"""
                <iframe 
                    src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" 
                    height="600px" 
                    type="application/pdf">
                </iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"PDF 渲染失败: {e}")
            st.info("💡 提示: 某些浏览器可能不支持内嵌 PDF 显示，请下载后使用本地阅读器。")

        # 模拟划词输入区 (因为浏览器内 PDF 交互限制)
        st.markdown("---")
        st.markdown("#### ✍️ 手动划词输入")
        st.info("💡 由于浏览器内 PDF 交互限制，请在此手动输入你在 PDF 中划选的文本。")

        selected_text = st.text_input("划选的术语/句子", key="pdf_selected_text")
        context = st.text_area("所在段落上下文", height=100, key="pdf_context")

        if st.button("📌 添加划词注释", key="add_annotation"):
            if selected_text:
                annotation = {
                    "text": selected_text,
                    "context": context,
                    "page": st.session_state.current_page,
                    "timestamp": st.datetime.now().isoformat()
                }
                st.session_state.annotations.append(annotation)
                st.success(f"✅ 已记录划词: {selected_text}")

                # 返回给主程序触发分析
                return annotation

    def _render_annotations_list(self):
        """渲染划词注释列表"""
        annotations = st.session_state.annotations

        if not annotations:
            return

        st.markdown("#### 📌 划词注释列表")

        for i, ann in enumerate(annotations):
            with st.expander(f"{i + 1}. {ann['text']} (第{ann['page']}页)", expanded=False):
                st.text(f"上下文: {ann['context']}")
                st.text(f"时间: {ann['timestamp']}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 分析此词", key=f"analyze_{i}"):
                        # 返回选中的注释用于分析
                        return ann
                with col2:
                    if st.button("🗑️ 删除", key=f"delete_{i}"):
                        st.session_state.annotations.pop(i)
                        st.rerun()

    def get_current_annotation(self) -> Optional[Dict[str, Any]]:
        """获取当前选中的划词注释"""
        return self._render_annotations_list()

    def add_annotation(self, text: str, context: str, page: int):
        """程序化添加划词注释"""
        annotation = {
            "text": text,
            "context": context,
            "page": page,
            "timestamp": st.datetime.now().isoformat()
        }
        st.session_state.annotations.append(annotation)