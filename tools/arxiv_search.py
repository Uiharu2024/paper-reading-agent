# tools/arxiv_search.py
"""
arXiv 预印本搜索工具
用于获取最新、最前沿的学术论文（特别是 CS、AI、数学领域）
"""

import arxiv
from langchain_core.tools import tool


@tool
def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    在 arXiv 数据库中搜索最新的预印本论文。
    适用于查找计算机科学、人工智能、数学等领域的最新前沿研究。

    Args:
        query: 搜索查询词 (支持 arXiv 高级语法，如 "ti:attention" 表示标题包含 attention)
        max_results: 返回的最大结果数 (1-5)
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=min(max(1, max_results), 5),
            sort_by=arxiv.SortCriterion.SubmittedDate,  # 按提交时间排序，获取最新
            sort_order=arxiv.SortOrder.Descending
        )

        results = list(client.results(search))

        if not results:
            return f"在 arXiv 中未找到与「{query}」相关的最新论文。"

        # 格式化输出
        output = f"arXiv 最新检索结果 (共 {len(results)} 篇)：\n\n"
        for i, paper in enumerate(results, 1):
            title = paper.title.replace("\n", " ")
            authors = ", ".join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."
            published = paper.published.strftime("%Y-%m-%d")
            summary = paper.summary.replace("\n", " ")
            if len(summary) > 200:
                summary = summary[:200] + "..."

            output += f"{i}. {title}\n"
            output += f"   作者: {authors} | 发布日期: {published}\n"
            output += f"   PDF链接: {paper.pdf_url}\n"
            output += f"   摘要: {summary}\n\n"

        return output.strip()

    except Exception as e:
        return f"arXiv 搜索出错: {str(e)}"