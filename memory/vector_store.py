# memory/vector_store.py
"""
向量数据库管理器 - 长期记忆存储

功能:
1. 存储用户的所有划词记录、解释、反馈
2. 支持语义相似度检索历史
3. 支持按论文、时间、类型等元数据过滤
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(
            self,
            collection_name: str = "reading_history",
            persist_directory: str = "./data/chroma_db",
            embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    ):
        """
        初始化向量存储

        Args:
            collection_name: Chroma 集合名称
            persist_directory: 持久化存储路径
            embedding_model: Embedding 模型路径 (HuggingFace 或本地路径)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 初始化 Embedding 模型
        # 如果使用 Qwen3-Embedding，需要从 HuggingFace 下载或使用本地路径
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if os.getenv("USE_GPU", "false") == "true" else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 初始化 Chroma 向量数据库
        from config.embedding_config import get_embeddings
        self.vectorstore = Chroma(
            collection_name="paper_annotations",
            embedding_function=get_embeddings(),
            persist_directory="./chroma_db"
        )

    def add_annotation(
            self,
            term: str,
            explanation: str,
            context: str,
            paper_id: str,
            paper_title: str,
            term_type: str,
            term_analysis: Dict[str, Any],
            page: int = 0,
            feedback: Optional[str] = None
    ) -> str:
        """
        添加一条划词记录到向量数据库

        Args:
            term: 划选的术语
            explanation: 生成的解释
            context: 上下文段落
            paper_id: 论文ID
            paper_title: 论文标题
            term_type: 术语类型
            term_analysis: 识别分析结果
            page: 页码
            feedback: 用户反馈 (positive/negative/None)

        Returns:
            记录的唯一ID
        """
        # 构建文档内容 (用于向量化)
        content = f"""术语: {term}
类型: {term_type}
标准名称: {term_analysis.get('standard_name_zh', '')} / {term_analysis.get('standard_name_en', '')}
解释: {explanation}
上下文: {context}"""

        # 构建元数据 (用于过滤)
        metadata = {
            "term": term,
            "term_type": term_type,
            "paper_id": paper_id,
            "paper_title": paper_title,
            "page": page,
            "timestamp": datetime.now().isoformat(),
            "standard_name_zh": term_analysis.get("standard_name_zh", ""),
            "standard_name_en": term_analysis.get("standard_name_en", ""),
            "domain_category": term_analysis.get("domain_category", ""),
            "has_feedback": feedback is not None,
            "feedback": feedback or ""
        }

        # 创建文档
        doc = Document(page_content=content, metadata=metadata)

        # 添加到向量数据库
        doc_id = self.vectorstore.add_documents([doc])[0]

        return doc_id

    def search_similar(
            self,
            query: str,
            k: int = 5,
            paper_id: Optional[str] = None,
            term_type: Optional[str] = None,
            min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        语义检索相似的历史划词记录

        Args:
            query: 查询文本 (通常是当前划词)
            k: 返回结果数量
            paper_id: 限定论文ID (同论文优先)
            term_type: 限定术语类型
            min_similarity: 最小相似度阈值

        Returns:
            相似记录列表，每条包含 content, metadata, score
        """
        # 构建过滤条件
        filter_dict = {}
        if paper_id:
            filter_dict["paper_id"] = paper_id
        if term_type:
            filter_dict["term_type"] = term_type

        # 执行相似度搜索
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_dict if filter_dict else None
        )

        # 格式化输出
        formatted_results = []
        for doc, score in results:
            if score >= min_similarity:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })

        return formatted_results

    def get_paper_history(self, paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取某篇论文的所有划词历史

        Args:
            paper_id: 论文ID
            limit: 最大返回数量

        Returns:
            历史记录列表
        """
        # Chroma 的 get 方法支持元数据过滤
        results = self.vectorstore.get(
            where={"paper_id": paper_id},
            limit=limit
        )

        history = []
        for i, doc_id in enumerate(results["ids"]):
            history.append({
                "id": doc_id,
                "content": results["documents"][i],
                "metadata": results["metadatas"][i]
            })

        return history

    def get_user_statistics(self) -> Dict[str, Any]:
        """
        获取用户的全局统计信息

        Returns:
            统计信息字典
        """
        # 获取所有记录
        all_docs = self.vectorstore.get()

        total_count = len(all_docs["ids"])

        # 统计术语类型分布
        type_distribution = {}
        paper_count = set()

        for metadata in all_docs["metadatas"]:
            term_type = metadata.get("term_type", "UNKNOWN")
            type_distribution[term_type] = type_distribution.get(term_type, 0) + 1
            paper_count.add(metadata.get("paper_id", ""))

        return {
            "total_annotations": total_count,
            "unique_papers": len(paper_count),
            "term_type_distribution": type_distribution,
            "avg_annotations_per_paper": total_count / max(len(paper_count), 1)
        }

    def delete_annotation(self, doc_id: str) -> bool:
        """
        删除指定的划词记录

        Args:
            doc_id: 记录ID

        Returns:
            是否删除成功
        """
        try:
            self.vectorstore.delete([doc_id])
            return True
        except Exception as e:
            print(f"Error deleting annotation {doc_id}: {e}")
            return False