# memory/user_profile.py
"""
用户画像管理器 - 个性化记忆

功能:
1. 基于划词历史动态推断用户背景
2. 评估用户对领域的熟悉程度
3. 识别用户的兴趣方向和解释偏好
"""

import os
from typing import Dict, Any, List, Optional
from collections import Counter
from datetime import datetime


class UserProfileManager:
    """用户画像管理器"""

    def __init__(self):
        """初始化用户画像"""
        self.profile = {
            "background": "unknown",  # 推断的学科背景
            "familiarity": 1,  # 领域熟悉度 (1-5)
            "interests": [],  # 兴趣方向列表
            "preferred_depth": "STANDARD",  # 偏好解释深度
            "language_preference": "zh",  # 语言偏好
            "total_annotations": 0,  # 总划词数
            "last_updated": datetime.now().isoformat()
        }

    def update_profile(
            self,
            annotation_history: List[Dict[str, Any]],
            current_term_analysis: Dict[str, Any],
            llm_client: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        基于划词历史更新用户画像

        Args:
            annotation_history: 历史划词记录
            current_term_analysis: 当前术语的识别分析
            llm_client: 可选的LLM客户端 (用于深度推断)

        Returns:
            更新后的用户画像
        """
        if not annotation_history:
            return self.profile

        # 1. 统计术语类型分布
        type_counter = Counter([
            h.get("term_type", "UNKNOWN")
            for h in annotation_history
        ])

        # 2. 统计领域分布
        domain_counter = Counter([
            h.get("term_analysis", {}).get("domain_category", "unknown")
            for h in annotation_history
            if h.get("term_analysis")
        ])

        # 3. 推断学科背景
        self.profile["background"] = self._infer_background(domain_counter)

        # 4. 评估领域熟悉度
        self.profile["familiarity"] = self._estimate_familiarity(
            annotation_history, type_counter
        )

        # 5. 识别兴趣方向
        self.profile["interests"] = self._identify_interests(domain_counter)

        # 6. 推断解释偏好
        self.profile["preferred_depth"] = self._infer_depth_preference(
            annotation_history, self.profile["familiarity"]
        )

        # 7. 更新统计信息
        self.profile["total_annotations"] = len(annotation_history)
        self.profile["last_updated"] = datetime.now().isoformat()

        return self.profile

    def _infer_background(self, domain_counter: Counter) -> str:
        """推断用户学科背景"""
        if not domain_counter:
            return "unknown"

        # 取最常见的领域
        most_common_domain = domain_counter.most_common(1)[0][0]

        # 映射到学科背景
        domain_mapping = {
            "computer_science": "计算机科学",
            "machine_learning": "机器学习",
            "natural_language_processing": "自然语言处理",
            "computer_vision": "计算机视觉",
            "bioinformatics": "生物信息学",
            "physics": "物理学",
            "mathematics": "数学",
            "economics": "经济学",
            "medicine": "医学"
        }

        return domain_mapping.get(most_common_domain, most_common_domain)

    def _estimate_familiarity(
            self,
            annotation_history: List[Dict[str, Any]],
            type_counter: Counter
    ) -> int:
        """
        评估用户对领域的熟悉程度 (1-5星)

        判断依据:
        - 划词总数 (越多越熟悉)
        - 术语类型分布 (METHOD/METRIC多说明深入阅读)
        - 是否查过基础概念 (CONCEPT多可能是初学者)
        """
        total = len(annotation_history)

        # 基础分: 基于划词数量
        if total < 5:
            base_score = 1
        elif total < 15:
            base_score = 2
        elif total < 30:
            base_score = 3
        elif total < 50:
            base_score = 4
        else:
            base_score = 5

        # 调整分: 基于术语类型
        concept_ratio = type_counter.get("CONCEPT", 0) / max(total, 1)
        method_ratio = type_counter.get("METHOD", 0) / max(total, 1)
        metric_ratio = type_counter.get("METRIC", 0) / max(total, 1)

        # 如果查了很多基础概念，可能是初学者
        if concept_ratio > 0.6:
            base_score = max(1, base_score - 1)

        # 如果查了很多方法和指标，说明深入阅读
        if method_ratio + metric_ratio > 0.4:
            base_score = min(5, base_score + 1)

        return base_score

    def _identify_interests(self, domain_counter: Counter) -> List[str]:
        """识别用户的兴趣方向 (Top 3领域)"""
        top_domains = domain_counter.most_common(3)
        return [domain for domain, count in top_domains]

    def _infer_depth_preference(
            self,
            annotation_history: List[Dict[str, Any]],
            familiarity: int
    ) -> str:
        """
        推断用户偏好的解释深度

        策略:
        - 初学者 (familiarity 1-2) → STANDARD (详细但通俗)
        - 中级 (familiarity 3) → STANDARD (平衡)
        - 专家 (familiarity 4-5) → QUICK (简洁精炼)
        """
        if familiarity <= 2:
            return "STANDARD"
        elif familiarity == 3:
            return "STANDARD"
        else:
            return "QUICK"

    def get_personalized_suggestions(self, current_term: str) -> Dict[str, Any]:
        """
        基于用户画像生成个性化建议

        Args:
            current_term: 当前查询的术语

        Returns:
            个性化建议字典
        """
        suggestions = {
            "explain_with_analogies": self.profile["familiarity"] <= 2,
            "connect_to_previous": self.profile["total_annotations"] > 3,
            "focus_on_application": "METHOD" in self.profile.get("interests", []),
            "provide_mathematical_detail": self.profile["familiarity"] >= 4
        }

        return suggestions

    def get_profile_summary(self) -> str:
        """获取用户画像的文本摘要 (用于注入Prompt)"""
        summary = f"""
用户学科背景: {self.profile['background']}
领域熟悉度: {'⭐' * self.profile['familiarity']} ({self.profile['familiarity']}/5)
主要兴趣方向: {', '.join(self.profile['interests'][:3])}
偏好解释深度: {self.profile['preferred_depth']}
总划词次数: {self.profile['total_annotations']}
"""
        return summary.strip()