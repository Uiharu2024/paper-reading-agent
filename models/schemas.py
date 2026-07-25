# models/schemas.py
"""
全局共享的 Pydantic 数据模型 (Schemas)
用于 API 接口定义、跨模块数据传递和类型校验
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== 🌐 API 交互模型 ====================

class ReadingRequest(BaseModel):
    """前端发起的划词阅读请求"""
    paper_id: str = Field(description="论文唯一标识符")
    paper_title: str = Field(description="论文标题")
    paper_domain: str = Field(default="通用学术", description="论文所属学科领域")
    selected_text: str = Field(description="用户划选的文本")
    selection_context: str = Field(description="划词所在的段落上下文")
    selection_page: int = Field(default=1, description="划词所在页码")

    # 可选：用户主动触发的控制参数
    force_deep_analysis: bool = Field(default=False, description="强制进行深度分析")
    user_feedback: Optional[str] = Field(default=None, description="用户对上一次解释的反馈")


class ReadingResponse(BaseModel):
    """返回给前端的完整阅读响应"""
    session_id: str = Field(description="本次会话/请求的唯一ID")
    term_analysis: Dict[str, Any] = Field(description="术语识别与分析结果")
    explanation: str = Field(description="生成的详细解释 (Markdown格式)")
    session_report: str = Field(description="即时知识卡片 (Markdown格式)")

    # 记忆与图谱数据 (用于前端渲染)
    related_terms: List[str] = Field(default_factory=list, description="相关推荐术语")
    knowledge_graph_snippet: Dict[str, Any] = Field(default_factory=dict, description="局部知识图谱数据")
    user_profile_summary: str = Field(default="", description="当前用户画像摘要")

    # 控制流状态
    processing_depth: str = Field(description="实际执行的处理深度")
    refinement_count: int = Field(default=0, description="优化迭代次数")


# ==================== 🧠 记忆系统数据模型 ====================

class AnnotationRecord(BaseModel):
    """单条划词记录的标准结构 (用于向量库和状态传递)"""
    term: str
    term_type: str
    standard_name_zh: str = ""
    standard_name_en: str = ""
    domain_category: str = ""
    explanation_summary: str = ""
    paper_id: str
    paper_title: str = ""
    page: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeEdge(BaseModel):
    """知识图谱中的边 (关系) 结构"""
    source: str = Field(description="源节点ID (通常是标准英文名或原文)")
    target: str = Field(description="目标节点ID")
    relation_type: str = Field(default="related", description="关系类型 (co_occurred, same_domain, parent, child等)")
    weight: float = Field(default=1.0, description="关系权重")
    paper_ids: List[str] = Field(default_factory=list, description="建立该关系的论文来源")


class UserProfileData(BaseModel):
    """用户画像的标准结构"""
    background: str = Field(default="unknown", description="推断的学科背景")
    familiarity: int = Field(default=1, ge=1, le=5, description="领域熟悉度 (1-5星)")
    interests: List[str] = Field(default_factory=list, description="兴趣方向列表")
    preferred_depth: str = Field(default="STANDARD", description="偏好解释深度 (QUICK/STANDARD/DEEP)")
    total_annotations: int = Field(default=0, description="总划词次数")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())