# memory/knowledge_graph.py
"""
知识图谱管理器 - 结构化记忆

功能:
1. 自动发现术语间的关联关系
2. 构建术语关系网络
3. 支持图谱查询和可视化导出
"""

import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
import os


class KnowledgeGraphManager:
    """知识图谱管理器"""

    def __init__(self, persist_path: str = "./data/knowledge_graph.json"):
        """
        初始化知识图谱

        Args:
            persist_path: 图谱持久化存储路径 (JSON格式)
        """
        self.persist_path = persist_path
        self.graph = nx.Graph()

        # 尝试从文件加载已有图谱
        self._load_from_disk()

    def add_term(
            self,
            term: str,
            term_type: str,
            paper_id: str,
            standard_name_zh: str = "",
            standard_name_en: str = "",
            domain_category: str = ""
    ) -> None:
        """
        添加术语节点

        Args:
            term: 术语原文
            term_type: 术语类型
            paper_id: 所属论文ID
            standard_name_zh: 标准中文名
            standard_name_en: 标准英文名
            domain_category: 学科领域
        """
        # 节点ID使用标准英文名(如果存在)，否则用原文
        node_id = standard_name_en.lower() if standard_name_en else term.lower()

        # 如果节点已存在，更新元数据
        if self.graph.has_node(node_id):
            # 追加论文ID (一个术语可能在多篇论文中出现)
            existing_papers = self.graph.nodes[node_id].get("paper_ids", [])
            if paper_id not in existing_papers:
                existing_papers.append(paper_id)
                self.graph.nodes[node_id]["paper_ids"] = existing_papers
        else:
            # 创建新节点
            self.graph.add_node(
                node_id,
                term=term,
                term_type=term_type,
                standard_name_zh=standard_name_zh,
                standard_name_en=standard_name_en,
                domain_category=domain_category,
                paper_ids=[paper_id],
                created_at=datetime.now().isoformat(),
                query_count=1
            )

    def add_edge(
            self,
            term1: str,
            term2: str,
            relation_type: str = "related",
            weight: float = 1.0,
            paper_id: str = ""
    ) -> None:
        """
        添加术语间的关系边

        Args:
            term1: 术语1 (节点ID)
            term2: 术语2 (节点ID)
            relation_type: 关系类型 (related/parent/child/uses/etc.)
            weight: 关系权重
            paper_id: 关系来源论文
        """
        term1_id = term1.lower()
        term2_id = term2.lower()

        # 确保两个节点都存在
        if not self.graph.has_node(term1_id) or not self.graph.has_node(term2_id):
            return

        # 如果边已存在，增加权重
        if self.graph.has_edge(term1_id, term2_id):
            current_weight = self.graph[term1_id][term2_id].get("weight", 1.0)
            self.graph[term1_id][term2_id]["weight"] = current_weight + weight

            # 追加论文来源
            papers = self.graph[term1_id][term2_id].get("paper_ids", [])
            if paper_id and paper_id not in papers:
                papers.append(paper_id)
                self.graph[term1_id][term2_id]["paper_ids"] = papers
        else:
            # 创建新边
            self.graph.add_edge(
                term1_id,
                term2_id,
                relation_type=relation_type,
                weight=weight,
                paper_ids=[paper_id] if paper_id else [],
                created_at=datetime.now().isoformat()
            )

    def discover_relations(
            self,
            current_term: str,
            annotation_history: List[Dict[str, Any]],
            window_size: int = 5
    ) -> List[Dict[str, str]]:
        """
        自动发现术语间的关联关系

        策略:
        1. 共现关系: 在最近的N次划词中共同出现的术语
        2. 语义关系: 通过向量相似度判断
        3. 层级关系: 通过术语类型和领域推断

        Args:
            current_term: 当前术语
            annotation_history: 历史划词记录
            window_size: 共现窗口大小

        Returns:
            发现的关系列表，每条包含 {source, target, relation_type, weight}
        """
        current_term_id = current_term.lower()
        discovered_edges = []

        # 获取最近N次划词记录
        recent_terms = [
            h.get("term", "").lower()
            for h in annotation_history[-window_size:]
            if h.get("term") and h.get("term").lower() != current_term_id
        ]

        # 1. 共现关系: 与最近查询的术语建立关联
        for related_term in set(recent_terms):
            if self.graph.has_node(related_term):
                discovered_edges.append({
                    "source": current_term_id,
                    "target": related_term,
                    "relation_type": "co_occurred",
                    "weight": 0.5
                })

        # 2. 同领域关系: 与同领域的其他术语建立关联
        current_node = self.graph.nodes.get(current_term_id, {})
        current_domain = current_node.get("domain_category", "")

        if current_domain:
            for node_id, node_data in self.graph.nodes(data=True):
                if (node_id != current_term_id and
                        node_data.get("domain_category") == current_domain):
                    discovered_edges.append({
                        "source": current_term_id,
                        "target": node_id,
                        "relation_type": "same_domain",
                        "weight": 0.3
                    })

        return discovered_edges

    def get_neighbors(
            self,
            term: str,
            depth: int = 2,
            min_weight: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        获取术语的邻居节点 (关联概念)

        Args:
            term: 术语 (节点ID)
            depth: 搜索深度 (1=直接邻居, 2=二度关联)
            min_weight: 最小边权重

        Returns:
            邻居节点列表
        """
        term_id = term.lower()

        if not self.graph.has_node(term_id):
            return []

        # BFS 搜索邻居
        neighbors = []
        visited = {term_id}
        queue = [(term_id, 0)]

        while queue:
            current, current_depth = queue.pop(0)

            if current_depth >= depth:
                continue

            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph[current][neighbor]

                # 过滤低权重边
                if edge_data.get("weight", 0) < min_weight:
                    continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    node_data = self.graph.nodes[neighbor]

                    neighbors.append({
                        "term_id": neighbor,
                        "term": node_data.get("term", neighbor),
                        "term_type": node_data.get("term_type", ""),
                        "relation_type": edge_data.get("relation_type", "related"),
                        "weight": edge_data.get("weight", 1.0),
                        "depth": current_depth + 1
                    })

                    queue.append((neighbor, current_depth + 1))

        # 按权重排序
        neighbors.sort(key=lambda x: x["weight"], reverse=True)
        return neighbors

    def get_subgraph(self, paper_id: str) -> Dict[str, Any]:
        """
        获取某篇论文相关的子图

        Args:
            paper_id: 论文ID

        Returns:
            子图的节点和边数据
        """
        # 筛选包含该论文ID的节点
        subgraph_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if paper_id in data.get("paper_ids", [])
        ]

        # 提取子图
        subgraph = self.graph.subgraph(subgraph_nodes)

        # 转换为可序列化格式
        nodes = []
        for node, data in subgraph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": data.get("term", node),
                "type": data.get("term_type", ""),
                "query_count": data.get("query_count", 1)
            })

        edges = []
        for source, target, data in subgraph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": data.get("relation_type", "related"),
                "weight": data.get("weight", 1.0)
            })

        return {"nodes": nodes, "edges": edges}

    def increment_query_count(self, term: str) -> None:
        """增加术语的查询次数"""
        term_id = term.lower()
        if self.graph.has_node(term_id):
            current_count = self.graph.nodes[term_id].get("query_count", 1)
            self.graph.nodes[term_id]["query_count"] = current_count + 1

    def _load_from_disk(self) -> None:
        """从磁盘加载图谱"""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
        except Exception as e:
            print(f"Warning: Failed to load knowledge graph: {e}")
            self.graph = nx.Graph()

    def save_to_disk(self) -> None:
        """持久化保存图谱"""
        try:
            # 转换为可序列化格式
            data = nx.node_link_data(self.graph)

            # 确保目录存在
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)

            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")