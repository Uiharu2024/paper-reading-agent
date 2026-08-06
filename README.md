# 📚 Paper Reading Agent

基于 LangGraph 的多智能体学术论文研读系统，自动化完成论文检索、摘要生成与知识图谱构建。

## ✨ 功能特性

- 🔍 **智能检索**：集成 arXiv / Semantic Scholar API，支持自然语言查询
- 🤖 **多Agent协作**：Router → Retriever → Explainer → Reporter 工作流
- 🧠 **混合记忆**：ChromaDB 向量检索 + Knowledge Graph 结构化知识
- 📊 **可视化界面**：Streamlit 交互式 UI，支持 PDF 原文对照阅读

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- Ollama（本地 LLM 推理）

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/你的用户名/paper-reading-agent.git
cd paper-reading-agent

# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 启动应用
streamlit run app.py
