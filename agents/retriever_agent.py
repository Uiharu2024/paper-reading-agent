# agents/retriever_agent.py
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph.state import PaperReadingState


# ===== 1. 定义检索工具 (Tools) =====
@tool
def search_paper_internal(query: str) -> str:
    """在论文全文中搜索相关内容，获取该术语在论文其他部分的提及和定义。"""
    # TODO: 实际项目中这里应调用 FAISS/Chroma 对论文分块进行向量检索
    # 这里作为 Mock 返回
    return f"[Mock 论文内部检索] 关于 '{query}'，在论文第3节提到...在第5节实验中也使用了该概念。"


@tool
def search_semantic_scholar(term: str, field: str) -> str:
    """在 Semantic Scholar 学术数据库中搜索该术语的相关论文和定义。"""
    # TODO: 实际项目中调用 Semantic Scholar API
    return f"[Mock Semantic Scholar] '{term}' 在 '{field}' 领域的经典文献包括: Paper A (2023), Paper B (2024)..."


@tool
def search_knowledge_base(query: str) -> str:
    """搜索预构建的通用学科知识库，获取基础概念解释。"""
    # TODO: 实际项目中查询本地向量数据库
    return f"[Mock 知识库] '{query}' 的基础定义是..."


tools = [search_paper_internal, search_semantic_scholar, search_knowledge_base]

# ===== 2. 初始化 LLM (Qwen3-8B, 绑定工具) =====
from config.llm_config import get_llm
llm_retriever = get_llm(temperature=0.1)

# 工具名称到工具对象的映射，用于执行
tool_map = {t.name: t for t in tools}


# ===== 3. Agent 节点函数 (内部 ReAct 循环) =====
def retriever_node(state: PaperReadingState) -> dict:
    """LangGraph 节点：多源信息检索 (ReAct 模式)"""
    term_analysis = state.get("term_analysis", {})
    term_en = term_analysis.get("standard_name_en", state.get("selected_text", ""))
    domain = term_analysis.get("domain_category", state.get("paper_domain", ""))

    # 构建初始消息
    messages = [
        HumanMessage(content=f"""你需要检索关于学术术语「{term_en}」(领域: {domain}) 的信息。
请根据需要使用提供的工具进行多源检索。
当收集到足够的信息后，请直接输出最终的检索结果摘要（JSON格式或纯文本），不要再调用工具。""")
    ]

    retrieved_info = []
    max_iterations = 4  # 限制最大工具调用轮数，防止死循环

    for _ in range(max_iterations):
        # 调用 LLM
        response: AIMessage = llm_retriever.invoke(messages)
        messages.append(response)

        # 如果没有工具调用，说明 LLM 认为检索完成
        if not response.tool_calls:
            # 提取最终的文本总结作为检索信息
            retrieved_info.append({"source": "llm_summary", "content": response.content})
            break

        # 执行工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 执行工具
            if tool_name in tool_map:
                tool_result = tool_map[tool_name].invoke(tool_args)
            else:
                tool_result = f"Error: Tool {tool_name} not found."

            # 记录结果
            retrieved_info.append({"source": tool_name, "args": tool_args, "content": tool_result})

            # 将工具结果加入消息历史
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

    return {
        "retrieved_info": retrieved_info,
        "current_agent": "retriever",
        "messages": [{"role": "retriever", "content": f"完成 {len(retrieved_info)} 条信息检索"}]
    }