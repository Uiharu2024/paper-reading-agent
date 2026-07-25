# memory/memory_manager.py
"""
记忆管理核心 - 统一接口层

整合向量存储、知识图谱、用户画像，为 LangGraph 节点提供统一的记忆操作接口
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from .vector_store import VectorStoreManager
from .knowledge_graph import KnowledgeGraphManager
from .user_profile import UserProfileManager


class MemoryManager:
    """记忆管理器 - 统一接口"""

    def __init__(self):
        """初始化三大记忆组件"""
        self.vector_store = VectorStoreManager()
        self.knowledge_graph = KnowledgeGraphManager()
        self.user_profile = UserProfileManager()

    async def update_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 LangGraph State 更新记忆系统

        这是 memory_updater_node 调用的核心方法

        Args:
            state: 当前 LangGraph 状态

        Returns:
            需要更新的状态字段
        """
        # 提取当前划词信息
        term = state.get("selected_text", "")
        explanation = state.get("explanation", "")
        context = state.get("selection_context", "")
        paper_id = state.get("paper_id", "")
        paper_title = state.get("paper_title", "")
        term_type = state.get("term_type", "")
        term_analysis = state.get("term_analysis", {})
        page = state.get("selection_page", 0)

        # 1. 更新向量数据库 (异步执行)
        await asyncio.to_thread(
            self.vector_store.add_annotation,
            term=term,
            explanation=explanation,
            context=context,
            paper_id=paper_id,
            paper_title=paper_title,
            term_type=term_type,
            term_analysis=term_analysis,
            page=page
        )

        # 2. 更新知识图谱
        standard_name_zh = term_analysis.get("standard_name_zh", "")
        standard_name_en = term_analysis.get("standard_name_en", "")

        self.knowledge_graph.add_term(
            term=term,
            term_type=term_type,
            paper_id=paper_id,
            standard_name_zh=standard_name_zh,
            standard_name_en=standard_name_en,
            domain_category=term_analysis.get("domain_category", "")
        )

        # 自动发现关系
        annotation_history = state.get("annotation_history", [])
        discovered_edges = self.knowledge_graph.discover_relations(
            current_term=standard_name_en or term,
            annotation_history=annotation_history
        )

        # 添加发现的关系
        for edge in discovered_edges:
            self.knowledge_graph.add_edge(
                term1=edge["source"],
                term2=edge["target"],
                relation_type=edge["relation_type"],
                weight=edge["weight"],
                paper_id=paper_id
            )

        # 增加查询次数
        self.knowledge_graph.increment_query_count(standard_name_en or term)

        # 持久化知识图谱
        self.knowledge_graph.save_to_disk()

        # 3. 更新用户画像
        self.user_profile.update_profile(
            annotation_history=annotation_history,
            current_term_analysis=term_analysis
        )

        # 4. 返回需要更新的状态字段
        return {
            "user_interest_profile": self.user_profile.profile,
            "knowledge_graph_edges": discovered_edges,
            "current_agent": "memory_updater",
            "messages": [{
                "role": "memory_updater",
                "content": f"记忆更新完成: 向量库+知识图谱+用户画像"
            }]
        }

    def get_personalized_context(self, state: Dict[str, Any]) -> str:
        """
        为 Explainer Agent 构建个性化上下文

        这是 explainer_agent 调用的核心方法，用于注入记忆

        Args:
            state: 当前状态

        Returns:
            格式化的个性化上下文字符串
        """
        current_term = state.get("selected_text", "")
        paper_id = state.get("paper_id", "")

        # 1. 从向量库检索相似历史
        similar_history = self.vector_store.search_similar(
            query=current_term,
            k=5,
            paper_id=paper_id,
            min_similarity=0.7
        )

        # 2. 从知识图谱获取关联概念
        term_analysis = state.get("term_analysis", {})
        standard_name = term_analysis.get("standard_name_en", current_term)
        neighbors = self.knowledge_graph.get_neighbors(
            term=standard_name,
            depth=2,
            min_weight=0.3
        )

        # 3. 获取用户画像摘要
        profile_summary = self.user_profile.get_profile_summary()

        # 4. 获取个性化建议
        suggestions = self.user_profile.get_personalized_suggestions(current_term)

        # 5. 构建完整上下文
        context = f"""
## 🧠 用户的阅读记忆

### 相似历史查询 (语义相关)
"""
        if similar_history:
            for i, hist in enumerate(similar_history[:3], 1):
                metadata = hist.get("metadata", {})
                context += f"{i}. **{metadata.get('term', '')}** ({metadata.get('term_type', '')}) - 相似度: {hist['similarity_score']:.2f}\n"
        else:
            context += "暂无相似历史记录\n"

        context += f"""
### 知识图谱关联概念
"""
        if neighbors:
            for neighbor in neighbors[:5]:
                context += f"- **{neighbor['term']}** ({neighbor['relation_type']}, 权重: {neighbor['weight']:.2f})\n"
        else:
            context += "暂无关联概念\n"

        context += f"""
### 用户画像
{profile_summary}

### 个性化建议
- 使用类比解释: {'是' if suggestions['explain_with_analogies'] else '否'}
- 关联历史概念: {'是' if suggestions['connect_to_previous'] else '否'}
- 侧重应用方法: {'是' if suggestions['focus_on_application'] else '否'}
- 提供数学细节: {'是' if suggestions['provide_mathematical_detail'] else '否'}
"""

        return context

    def get_paper_knowledge_graph(self, paper_id: str) -> Dict[str, Any]:
        """
        获取某篇论文的知识图谱数据 (用于UI可视化)

        Args:
            paper_id: 论文ID

        Returns:
            知识图谱的节点和边数据
        """
        return self.knowledge_graph.get_subgraph(paper_id)

    def get_user_statistics(self) -> Dict[str, Any]:
        """获取用户的全局统计信息"""
        return self.vector_store.get_user_statistics()