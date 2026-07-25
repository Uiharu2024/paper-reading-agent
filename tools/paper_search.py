# tools/paper_search.py
"""
论文内部语义搜索工具 (RAG)
使用 FAISS 构建论文分块的向量索引，支持局部语义检索
"""

import os
import hashlib
from typing import Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 全局缓存：避免对同一篇论文重复构建向量索引
_paper_index_cache = {}


def _get_embeddings():
    """获取 Embedding 模型 (单例)"""
    # 实际项目中应从 config 读取模型路径
    model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda' if os.getenv("USE_GPU") == "true" else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def _build_paper_index(paper_id: str, paper_text: str) -> FAISS:
    """构建或获取论文的 FAISS 索引"""
    if paper_id in _paper_index_cache:
        return _paper_index_cache[paper_id]

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " "]
    )
    chunks = text_splitter.split_text(paper_text)
    documents = [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

    # 构建 FAISS 索引
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    _paper_index_cache[paper_id] = vectorstore
    return vectorstore


@tool
def search_paper_internal(query: str, paper_id: str = "default", paper_text: str = "") -> str:
    """
    在论文全文中搜索相关内容。
    用于查找该术语在论文其他章节（如引言、方法、实验）中的提及、定义或具体应用。

    Args:
        query: 搜索查询词（通常是术语名称或相关描述）
        paper_id: 论文的唯一标识符（用于缓存索引）
        paper_text: 论文的纯文本全文（首次查询时需提供，后续可省略）
    """
    try:
        # 1. 获取或构建索引
        if paper_id not in _paper_index_cache and not paper_text:
            return "错误：未找到该论文的缓存索引，且未提供论文全文。"

        if paper_id not in _paper_index_cache:
            _build_paper_index(paper_id, paper_text)

        vectorstore = _paper_index_cache[paper_id]

        # 2. 执行相似度搜索
        docs = vectorstore.similarity_search(query, k=3)

        if not docs:
            return f"在论文中未找到与「{query}」相关的内容。"

        # 3. 格式化输出
        result = f"在论文中找到 {len(docs)} 处与「{query}」相关的内容：\n\n"
        for i, doc in enumerate(docs, 1):
            # 截取前200字符避免过长
            content = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            result += f"[片段 {i}] (Chunk ID: {doc.metadata.get('chunk_id', 'N/A')})\n{content}\n\n"

        return result.strip()

    except Exception as e:
        return f"论文内部搜索出错: {str(e)}"