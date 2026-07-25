# graph/conditions.py
"""
LangGraph 条件边函数集合
决定工作流的动态走向：路由分发 & 优化循环判断
"""

from .state import PaperReadingState

# 最大优化重试次数，防止死循环
MAX_REFINEMENT_ITERATIONS = 2


def route_after_router(state: PaperReadingState) -> str:
    """
    条件边1: Router之后的路由分发

    根据 processing_depth 决定后续流程:
    - QUICK → 跳过检索，直接生成简单解释
    - STANDARD → 走完整流水线
    - DEEP → 走完整流水线 + 标记需要优化

    Returns:
        下一个节点的名称字符串
    """
    depth = state.get("processing_depth", "STANDARD")
    term_type = state.get("term_type", "CONCEPT")

    # 普通词汇或公式 → 快速通道
    if term_type in ("GENERAL", "FORMULA") or depth == "QUICK":
        return "explainer_fast"

    # 标准/深度处理 → 完整流水线
    return "recognizer"


def should_refine(state: PaperReadingState) -> str:
    """
    条件边2: Memory Updater之后的优化循环判断

    判断逻辑:
    1. 如果 needs_refinement=True 且未超过最大迭代次数 → 回到retriever重新检索
    2. 否则 → 结束本次划词处理

    触发优化的场景:
    - Router判定为DEEP深度
    - Explainer自评置信度不足
    - 用户在UI点击了「🔄 更深入」按钮

    Returns:
        "refine" 或 "done"
    """
    needs_refinement = state.get("needs_refinement", False)
    refinement_count = state.get("refinement_count", 0)

    if needs_refinement and refinement_count < MAX_REFINEMENT_ITERATIONS:
        return "refine"

    return "done"


def route_after_memory_updater(state: PaperReadingState) -> str:
    """
    条件边3: Memory Updater之后统一出口

    整合优化判断与报告生成逻辑:
    - 需要优化 → refinement_check节点
    - 不需要优化 → END
    """
    return should_refine(state)