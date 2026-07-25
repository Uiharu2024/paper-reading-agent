# tools/scholar_api.py
"""
Semantic Scholar 学术搜索工具
无需 API Key 即可使用基础功能，但需注意 Rate Limit (100 req/5min)
"""

import time
import requests
from typing import Optional
from langchain_core.tools import tool

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


@tool
def search_semantic_scholar(term: str, field_of_study: Optional[str] = None, limit: int = 3) -> str:
    """
    在 Semantic Scholar 学术数据库中搜索与术语相关的权威论文。
    用于获取该术语的经典文献、最新研究进展和权威定义。

    Args:
        term: 搜索的学术术语或关键词
        field_of_study: 限定学科领域 (如 Computer Science, Medicine)，可选
        limit: 返回的论文数量 (1-5)
    """

    params = {
        "query": term,
        "limit": min(max(1, limit), 5),
        "fields": "title,authors,year,abstract,citationCount,venue"
    }

    if field_of_study:
        params["fieldsOfStudy"] = field_of_study

    headers = {
        "Accept": "application/json",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=10)

            # 处理 Rate Limit (HTTP 429)
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            papers = data.get("data", [])
            if not papers:
                return f"在 Semantic Scholar 中未找到与「{term}」相关的论文。"

            # 格式化输出
            result = f"Semantic Scholar 检索结果 (共 {len(papers)} 篇)：\n\n"
            for i, paper in enumerate(papers, 1):
                title = paper.get("title", "Unknown Title")
                year = paper.get("year", "N/A")
                venue = paper.get("venue", "N/A")
                citations = paper.get("citationCount", 0)
                authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:3]])
                if len(paper.get("authors", [])) > 3:
                    authors += " et al."
                abstract = paper.get("abstract", "No abstract available.")
                if abstract and len(abstract) > 200:
                    abstract = abstract[:200] + "..."

                result += f"{i}. {title} ({year}, {venue})\n"
                result += f"   作者: {authors} | 引用数: {citations}\n"
                result += f"   摘要: {abstract}\n\n"

            return result.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                return f"Semantic Scholar API 请求失败: {str(e)}"
            time.sleep(2)

    return "Semantic Scholar 搜索超时或失败。"