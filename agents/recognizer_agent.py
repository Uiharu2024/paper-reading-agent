# agents/recognizer_agent.py
import os
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import PaperReadingState


class TermRecognition(BaseModel):
    """术语识别结构化输出"""
    standard_name_zh: str = Field(description="标准中文名称")
    standard_name_en: str = Field(description="标准英文名称")
    domain_category: str = Field(description="所属细分学科领域")
    context_meaning: str = Field(description="该术语在当前论文上下文中的具体含义")
    relevance_score: int = Field(description="与论文核心主题的关联度 (1-5)", ge=1, le=5)
    related_terms: List[str] = Field(description="基于上下文推荐的相关术语列表(3-5个)")


# 开启轻量思考
from config.llm_config import get_llm
llm_recognizer = get_llm(temperature=0.1, structured_output=TermRecognition)

recognizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是{paper_domain}领域的学术术语专家。
请对用户在论文中划选的词汇进行精确识别和归一化分析。
注意：某些词汇在不同领域含义不同，请务必结合提供的「论文上下文」进行判断。"""),
    ("human", """论文标题：{paper_title}
上下文段落：
{selection_context}

用户划词：「{selected_text}」
术语类型预判：{term_type}

请输出结构化的识别结果。""")
])


def recognizer_node(state: PaperReadingState) -> dict:
    """LangGraph 节点：术语识别"""
    chain = recognizer_prompt | llm_recognizer

    result: TermRecognition = chain.invoke({
        "paper_domain": state.get("paper_domain", "通用"),
        "paper_title": state.get("paper_title", "未知论文"),
        "selection_context": state.get("selection_context", ""),
        "selected_text": state.get("selected_text", ""),
        "term_type": state.get("term_type", "CONCEPT")
    })

    # 将结果序列化为字典存入状态
    term_analysis = result.model_dump()

    return {
        "term_analysis": term_analysis,
        "current_agent": "recognizer",
        "messages": [
            {"role": "recognizer", "content": f"识别完成: {result.standard_name_zh} ({result.standard_name_en})"}]
    }