# graph/workflow.py
"""
论文阅读多智能体系统 - LangGraph 工作流编排
"""

import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from .state import PaperReadingState
from .conditions import route_after_router, route_after_memory_updater


def build_graph(use_memory_saver=True):
    """
    构建并返回编译后的 LangGraph 应用

    工作流拓扑:
        START → router ──┬──(QUICK)──→ explainer_fast → reporter → memory_updater ─┐
                         │                                                          │
                         └──(STANDARD/DEEP)→ recognizer → retriever → explainer     │
                                                                              ↓     │
                                                              reporter → memory_updater
                                                                              ↓
                                                                    [should_refine?]
                                                                   ↙              ↘
                                                              (refine)           (done)
                                                           retriever               END
    """
    # ✅ 修复1: 延迟导入打破循环依赖
    # ✅ 修复2: 补全缺失的 recognizer_node 导入
    from agents import (
        router_node,
        recognizer_node,
        retriever_node,
        explainer_node,
        reporter_node,
    )

    # ✅ 修复3: 移除多余的 graph = StateGraph(...)，只保留 workflow
    workflow = StateGraph(PaperReadingState)

    # ========== 1. 注册所有节点 ==========
    workflow.add_node("router", router_node)
    workflow.add_node("recognizer", recognizer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("explainer", explainer_node)
    workflow.add_node("reporter", reporter_node)
    workflow.add_node("memory_updater", _memory_updater_node)

    # 可选: 轻量解释节点(QUICK路径专用，使用更小模型)
    workflow.add_node("explainer_fast", explainer_node)

    # ========== 2. 定义边 ==========
    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "recognizer": "recognizer",
            "explainer_fast": "explainer_fast",
        },
    )

    workflow.add_edge("recognizer", "retriever")
    workflow.add_edge("retriever", "explainer")
    workflow.add_edge("explainer", "reporter")
    workflow.add_edge("explainer_fast", "reporter")
    workflow.add_edge("reporter", "memory_updater")

    workflow.add_conditional_edges(
        "memory_updater",
        route_after_memory_updater,
        {
            "refine": "retriever",
            "done": END,
        },
    )

    # ========== 3. 编译图(带持久化Checkpointer) ==========
    if use_memory_saver:
        checkpointer = MemorySaver()
    else:
        # 生产环境使用异步 SQLite
        raise NotImplementedError("AsyncSqliteSaver 需要在 async with 中使用")

    app = workflow.compile(checkpointer=checkpointer)
    return app


async def _memory_updater_node(state: PaperReadingState) -> dict:
    """
    记忆更新包装节点
    将 memory_manager 的异步操作封装为 LangGraph 节点。
    """
    from memory.memory_manager import MemoryManager

    manager = MemoryManager()
    updates = await manager.update_from_state(state)

    if state.get("needs_refinement", False):
        updates["refinement_count"] = state.get("refinement_count", 0) + 1
        if updates["refinement_count"] >= 2:
            updates["needs_refinement"] = False

    updates["current_agent"] = "memory_updater"
    return updates


# ========== 模块级单例(避免重复编译) ==========
_graph_instance = None


def get_graph():
    """获取全局唯一的编译后Graph实例"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance