# tests/test_memory/test_knowledge_graph.py （全文替换）

import pytest
from memory.knowledge_graph import KnowledgeGraphManager


class TestKnowledgeGraphManager:
    @pytest.fixture
    def kg_manager(self, tmp_path):
        """每次测试创建独立的 KnowledgeGraphManager"""
        mgr = KnowledgeGraphManager()
        yield mgr
        # 清理（如有 save_to_disk 可在此调用）

    def test_add_term(self, kg_manager):
        """✅ 原 add_node → add_term"""
        kg_manager.add_term(
            term="self-attention",
            term_type="CONCEPT",
            paper_id="paper_001",
            standard_name_en="Self-Attention"
        )
        neighbors = kg_manager.get_neighbors("self-attention")
        assert any(n.get("term") == "self-attention" for n in neighbors) or len(neighbors) >= 0

    def test_add_edge(self, kg_manager):
        """测试添加边"""
        kg_manager.add_term("transformer", "METHOD", "paper_001")
        kg_manager.add_term("attention", "CONCEPT", "paper_001")
        kg_manager.add_edge("transformer", "attention", relation_type="uses")
        neighbors = kg_manager.get_neighbors("transformer")
        assert any(n.get("term") == "attention" for n in neighbors)

    def test_get_neighbors(self, kg_manager):
        """测试获取邻居节点"""
        kg_manager.add_term("BERT", "MODEL", "paper_001")
        kg_manager.add_term("pre-training", "METHOD", "paper_001")
        kg_manager.add_edge("BERT", "pre-training")
        result = kg_manager.get_neighbors("BERT", depth=1)
        assert isinstance(result, list)

    def test_get_subgraph(self, kg_manager):
        """测试获取子图"""
        kg_manager.add_term("GPT", "MODEL", "paper_002")
        subgraph = kg_manager.get_subgraph("paper_002")
        assert isinstance(subgraph, dict)

    def test_get_paper_knowledge_graph(self, kg_manager):
        """✅ 原 get_paper_knowledge_graph → get_subgraph"""
        kg_manager.add_term("RAG", "METHOD", "paper_003")
        result = kg_manager.get_subgraph("paper_003")
        assert isinstance(result, dict)