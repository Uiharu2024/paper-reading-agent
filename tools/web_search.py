# tools/web_search.py
"""
通用网络搜索工具 (DuckDuckGo)
作为兜底方案，用于搜索维基百科、技术博客、百科等通用网页信息
"""

from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def search_web(query: str, max_results: int = 3) -> str:
    """
    使用 DuckDuckGo 搜索引擎在互联网上搜索通用信息。
    适用于查找基础概念的通俗解释、维基百科定义、技术博客教程等。

    Args:
        query: 搜索关键词
        max_results: 返回的网页数量 (1-5)
    """
    try:
        with DDGS() as ddgs:
            # 搜索网页
            results = [r for r in ddgs.text(query, max_results=min(max(1, max_results), 5))]

            if not results:
                return f"在互联网上未找到与「{query}」相关的信息。"

            # 格式化输出
            output = f"网络搜索结果 (共 {len(results)} 条)：\n\n"
            for i, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                href = r.get("href", "#")
                body = r.get("body", "No description.")

                output += f"{i}. {title}\n"
                output += f"   链接: {href}\n"
                output += f"   摘要: {body}\n\n"

            return output.strip()

    except Exception as e:
        return f"网络搜索出错 (可能是请求频率过高): {str(e)}"