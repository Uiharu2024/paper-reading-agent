# agents/reporter_agent.py
import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import PaperReadingState

# 适合长文本总结和报告生成
from config.llm_config import get_llm
llm_reporter = get_llm(temperature=0.4)

# 即时报告 Prompt
instant_report_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个学术笔记整理专家。请将以下术语解释转化为一张结构化的「知识卡片」(Markdown格式)。"),
    ("human", """术语：{selected_text}
识别信息：{term_analysis}
详细解释：{explanation}

请输出包含以下模块的知识卡片：
1. 📌 核心定义 (一句话)
2. 📖 详细解析 (保留原解释的核心内容，精简排版)
3. 🔗 知识关联 (列出相关概念)
4. 💡 记忆锚点 (提供一个方便记忆的口诀或类比)
""")
])

# 累积报告 Prompt
cumulative_report_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个学术阅读导师。根据学生在一篇论文中的所有划词记录，生成一份阶段性的「阅读分析与知识图谱报告」。"),
    ("human", """论文标题：{paper_title}
论文领域：{paper_domain}
历史划词记录 (按时间顺序)：
{history_json}

请生成一份 Markdown 格式的综合报告，包含：
1. 📊 **阅读画像**：分析学生目前的关注点和知识盲区。
2. 🕸️ **概念网络**：梳理这些划词术语之间的逻辑关系（如：A是B的基础，C用于优化A）。
3. 🎯 **核心主线**：用一段话总结这些术语是如何串联起这篇论文的核心思想的。
4. 📚 **下一步建议**：推荐接下来应该重点阅读的章节或需要补充的前置知识。
""")
])


def reporter_node(state: PaperReadingState) -> dict:
    """LangGraph 节点：报告生成"""
    history = state.get("annotation_history", [])
    history_count = len(history)

    updates = {"current_agent": "reporter"}

    # 1. 每次都生成即时知识卡片
    chain_instant = instant_report_prompt | llm_reporter
    instant_report = chain_instant.invoke({
        "selected_text": state.get("selected_text", ""),
        "term_analysis": json.dumps(state.get("term_analysis", {}), ensure_ascii=False),
        "explanation": state.get("explanation", "")
    })
    updates["session_report"] = instant_report.content

    # 2. 每累积 5 次划词，生成一次阶段性综合报告
    if history_count > 0 and history_count % 5 == 0:
        chain_cumulative = cumulative_report_prompt | llm_reporter
        cumulative_report = chain_cumulative.invoke({
            "paper_title": state.get("paper_title", "未知论文"),
            "paper_domain": state.get("paper_domain", "通用"),
            "history_json": json.dumps(history, ensure_ascii=False, indent=2)
        })
        updates["cumulative_report"] = cumulative_report.content
        updates["messages"] = [
            {"role": "reporter", "content": f"生成即时卡片，并触发第 {history_count} 次阶段性综合报告。"}]
    else:
        updates["messages"] = [{"role": "reporter", "content": "生成即时知识卡片。"}]

    # 记录本次划词到历史中 (利用 operator.add 自动追加)
    new_annotation = {
        "term": state.get("selected_text"),
        "type": state.get("term_type"),
        "timestamp": datetime.now().isoformat(),
        "explanation_summary": state.get("explanation", "")[:100] + "..."
    }
    updates["annotation_history"] = [new_annotation]

    return updates