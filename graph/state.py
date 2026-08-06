# graph/state.py
"""
论文阅读多智能体系统 - 全局状态定义
使用 TypedDict + Annotated 实现类型安全与自动归约
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """自定义归约函数：深度合并字典，右侧覆盖左侧同名键"""
    merged = left.copy()
    merged.update(right)
    return merged


class PaperReadingState(TypedDict):
    # ==================== 📥 输入与上下文 ====================
    paper_id: str  # 论文唯一标识符
    paper_title: str  # 论文标题
    paper_domain: str  # 所属学科领域
    paper_full_text: str  # 论文全文(用于内部RAG)
    selected_text: str  # 用户当前划选的文本
    selection_context: str  # 划词所在段落上下文
    selection_page: int  # 划词所在页码

    # ==================== ⚙️ Agent 处理中间态 ====================
    term_type: str  # 术语类型(CONCEPT/METHOD/METRIC等)
    processing_depth: str  # 处理深度(QUICK/STANDARD/DEEP)
    term_analysis: Dict[str, Any]  # 识别Agent的结构化分析结果
    retrieved_info: List[Dict[str, Any]]  # 检索Agent返回的多源信息列表
    explanation: str  # 解释Agent生成的详细解释文本

    # ==================== 🧠 记忆与累积状态 ====================
    # operator.add: 更新时自动追加到列表末尾
    annotation_history: Annotated[List[Dict[str, Any]], operator.add]
    messages: Annotated[List[BaseMessage], operator.add]  # Agent间通信消息

    # merge_dicts: 每次更新时深度合并，保留历史画像字段
    user_interest_profile: Annotated[Dict[str, Any], merge_dicts]
    knowledge_graph_edges: Annotated[List[Dict[str, str]], operator.add]

    # ==================== 📤 输出结果 ====================
    session_report: str  # 当次即时知识卡片(Markdown)
    cumulative_report: str  # 阶段性综合分析报告(Markdown)

    # ==================== 🎛️ 控制流 ====================
    current_agent: str  # 正在执行的Agent节点名
    needs_refinement: bool  # 是否需要进入优化循环
    refinement_count: int  # 当前已优化的次数
