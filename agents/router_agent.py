# agents/router_agent.py
import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 假设状态定义在 graph.state 中
from graph.state import PaperReadingState


# ===== 1. 定义结构化输出模型 =====
class RouterDecision(BaseModel):
    """路由决策的结构化输出"""
    term_type: str = Field(
        description="术语类型: CONCEPT(概念), METHOD(方法), METRIC(指标), REFERENCE(引用), FORMULA(公式), GENERAL(普通词汇)",
        enum=["CONCEPT", "METHOD", "METRIC", "REFERENCE", "FORMULA", "GENERAL"]
    )
    processing_depth: str = Field(
        description="处理深度: QUICK(快速), STANDARD(标准), DEEP(深度)",
        enum=["QUICK", "STANDARD", "DEEP"]
    )
    brief_reasoning: str = Field(description="简短的路由判断理由，不超过20字")


# ===== 2. 初始化 LLM =====
from config.llm_config import get_llm
llm_router = get_llm(temperature=0.1, structured_output=RouterDecision)

# ===== 3. 定义 Prompt =====
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个学术论文阅读助手的调度模块。
根据用户划选的文本和论文上下文，快速判断术语类型和所需的处理深度。
论文领域：{paper_domain}
历史划词摘要：{history_summary}"""),
    ("human", """当前划词：「{selected_text}」
所在上下文段落：
{selection_context}

请输出你的路由决策。""")
])


# ===== 4. Agent 节点函数 =====
def router_node(state: PaperReadingState) -> dict:
    """
    LangGraph 节点：路由调度
    """
    selected_text = state.get("selected_text", "")
    selection_context = state.get("selection_context", "")
    paper_domain = state.get("paper_domain", "通用学术")

    # 简单摘要历史划词，避免 Prompt 过长
    history = state.get("annotation_history", [])
    history_summary = ", ".join([h.get("term", "") for h in history[-5:]]) if history else "无"

    # 构建 Chain 并调用
    chain = router_prompt | llm_router
    decision: RouterDecision = chain.invoke({
        "selected_text": selected_text,
        "selection_context": selection_context,
        "paper_domain": paper_domain,
        "history_summary": history_summary
    })

    # 返回状态更新
    return {
        "term_type": decision.term_type,
        "processing_depth": decision.processing_depth,
        "current_agent": "router",
        "messages": [{"role": "router",
                      "content": f"路由决策: {decision.term_type} / {decision.processing_depth} ({decision.brief_reasoning})"}]
    }