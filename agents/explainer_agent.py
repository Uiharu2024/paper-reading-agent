# agents/explainer_agent.py
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import PaperReadingState

# 开启深度思考模式
from config.llm_config import get_llm
llm_explainer = get_llm(temperature=0.5)
explainer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深的 {paper_domain} 领域学者，正在指导一位学生阅读学术论文。
你的任务是基于检索到的资料，为学生详细、通俗地解释论文中的某个术语。

## 解释要求：
1. **一句话定义**：用最精炼的语言概括核心含义。
2. **详细解释**：结合论文上下文，说明该术语的具体机制或原理。
3. **本文作用**：说明该术语在这篇论文中扮演什么角色（为什么作者要用它）。
4. **概念关联**：结合学生的【阅读记忆】，用他之前查过的概念来做类比或建立联系。
5. **通俗类比**：给出一个生活中的例子帮助理解。

请使用 Markdown 格式输出，结构清晰。"""),
    ("human", """## 当前术语信息
- 划词原文：{selected_text}
- 识别结果：{term_analysis}
- 所在上下文：{selection_context}

## 检索到的参考资料
{retrieved_info}

## 🧠 学生的阅读记忆与画像 (重要！请据此调整解释策略)
{memory_context}

请开始你的解释：""")
])


def _build_memory_context(state: PaperReadingState) -> str:
    """辅助函数：从状态中提取并格式化记忆上下文"""
    history = state.get("annotation_history", [])
    profile = state.get("user_interest_profile", {})

    if not history and not profile:
        return "该学生是首次使用本系统，暂无历史记忆，请使用通用且详尽的解释方式。"

    # 提取最近查过的关联术语
    recent_terms = [h.get("term", "") for h in history[-5:]]

    context = f"""
- **用户推断背景**：{profile.get('background', '未知')}
- **领域熟悉度**：{profile.get('familiarity', '初学者')} (1-5星)
- **最近查询过的术语**：{', '.join(recent_terms)}
- **个性化建议**：尽量将当前术语与上述「最近查询过的术语」建立逻辑联系，帮助学生构建知识网络。
"""
    return context


def explainer_node(state: PaperReadingState) -> dict:
    """LangGraph 节点：生成个性化解释"""

    # 将检索信息格式化为易读的文本
    retrieved_info = state.get("retrieved_info", [])
    info_text = "\n\n".join([
        f"[来源: {info.get('source', 'unknown')}]\n{info.get('content', '')}"
        for info in retrieved_info
    ])

    chain = explainer_prompt | llm_explainer

    # 使用流式或普通调用，这里使用普通调用获取完整文本
    response = chain.invoke({
        "paper_domain": state.get("paper_domain", "通用"),
        "selected_text": state.get("selected_text", ""),
        "term_analysis": json.dumps(state.get("term_analysis", {}), ensure_ascii=False, indent=2),
        "selection_context": state.get("selection_context", ""),
        "retrieved_info": info_text,
        "memory_context": _build_memory_context(state)
    })

    return {
        "explanation": response.content,
        "current_agent": "explainer",
        "messages": [{"role": "explainer", "content": "解释生成完毕"}]
    }