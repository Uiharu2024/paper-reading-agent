# ui/knowledge_graph_viz.py
"""
知识图谱可视化组件

功能:
1. 渲染术语关系网络图
2. 支持交互式探索 (点击节点查看详情)
3. 按论文或领域过滤子图
"""

import streamlit as st
import networkx as nx
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KnowledgeGraphVisualizer:
    """知识图谱可视化器"""

    def __init__(self):
        """初始化可视化器"""
        if "selected_node" not in st.session_state:
            st.session_state.selected_node = None

    def render(self, graph_data: Dict[str, Any], title: str = "🕸️ 知识图谱"):
        """
        渲染知识图谱可视化

        Args:
            graph_data: 包含 nodes 和 edges 的字典
            title: 图表标题
        """
        st.subheader(title)

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            st.info("📭 暂无知识图谱数据。多查询几个术语，AI 会自动构建概念关联网络。")
            return

        # 1. 统计信息
        self._render_stats(nodes, edges)

        # 2. 网络图可视化
        self._render_network_graph(nodes, edges)

        # 3. 节点详情
        self._render_node_details(nodes)

    def _render_stats(self, nodes: List[Dict], edges: List[Dict]):
        """渲染图谱统计信息"""
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("概念节点数", len(nodes))
        with col2:
            st.metric("关系边数", len(edges))
        with col3:
            # 计算平均连接度
            avg_degree = (2 * len(edges)) / max(len(nodes), 1)
            st.metric("平均连接度", f"{avg_degree:.2f}")

    def _render_network_graph(self, nodes: List[Dict], edges: List[Dict]):
        """渲染交互式网络图"""
        # 使用 Plotly 绘制网络图
        try:
            # 构建 NetworkX 图
            G = nx.Graph()

            # 添加节点
            for node in nodes:
                G.add_node(
                    node["id"],
                    label=node.get("label", node["id"]),
                    type=node.get("type", "UNKNOWN"),
                    query_count=node.get("query_count", 1)
                )

            # 添加边
            for edge in edges:
                G.add_edge(
                    edge["source"],
                    edge["target"],
                    type=edge.get("type", "related"),
                    weight=edge.get("weight", 1.0)
                )

            # 使用 spring layout 计算节点位置
            pos = nx.spring_layout(G, k=2, iterations=50)

            # 准备 Plotly 数据
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x = []
            node_y = []
            node_text = []
            node_colors = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                node_data = G.nodes[node]
                node_text.append(
                    f"<b>{node_data['label']}</b><br>"
                    f"类型: {node_data['type']}<br>"
                    f"查询次数: {node_data['query_count']}"
                )

                # 根据类型分配颜色
                color_map = {
                    "CONCEPT": "#636EFA",
                    "METHOD": "#EF553B",
                    "METRIC": "#00CC96",
                    "UNKNOWN": "#AB63FA"
                }
                node_colors.append(color_map.get(node_data["type"], "#AB63FA"))

            # 创建 Plotly 图表
            fig = go.Figure()

            # 添加边
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none'
            ))

            # 添加节点
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=[15 + G.nodes[n]['query_count'] * 2 for n in G.nodes()],
                    color=node_colors,
                    line=dict(width=2, color='white')
                ),
                text=[G.nodes[n]['label'] for n in G.nodes()],
                textposition="top center",
                textfont=dict(size=10),
                hovertext=node_text,
                hoverinfo='text'
            ))

            # 更新布局
            fig.update_layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"图谱可视化失败: {e}")
            # 降级方案：显示表格
            st.dataframe(nodes)
            st.dataframe(edges)

    def _render_node_details(self, nodes: List[Dict]):
        """渲染节点详情列表"""
        st.markdown("#### 📋 概念节点列表")

        # 按查询次数排序
        sorted_nodes = sorted(nodes, key=lambda x: x.get("query_count", 1), reverse=True)

        for node in sorted_nodes:
            with st.expander(f"{node['label']} ({node.get('type', 'UNKNOWN')})", expanded=False):
                st.text(f"节点 ID: {node['id']}")
                st.text(f"查询次数: {node.get('query_count', 1)}")

                if st.button("🔍 重新分析此概念", key=f"reanalyze_{node['id']}"):
                    st.session_state.selected_node = node
                    st.rerun()

    def get_selected_node(self) -> Optional[Dict[str, Any]]:
        """获取用户选中的节点 (用于触发重新分析)"""
        return st.session_state.selected_node