from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation


class CommunityGraph(nx.Graph):
    """
    Enhanced graph implementation for community detection and analysis.

    This class extends NetworkX Graph to provide sophisticated community
    detection capabilities with multiple algorithms and visualization tools.
    """

    def __init__(
        self,
        base_graph: Optional[nx.Graph] = None,
        vertices: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        drop_unconnected: bool = True,
    ) -> None:
        super().__init__()
        self._drop_unconnected = drop_unconnected

        if base_graph is not None:
            if drop_unconnected:
                base_graph = base_graph.copy()
                base_graph.remove_nodes_from(
                    [n for n, d in nx.degree(base_graph) if d == 0]
                )
            self.add_nodes_from(base_graph.nodes)
            self.add_edges_from(base_graph.edges.data())
        else:
            if vertices is not None:
                self.add_nodes_from(vertices)
            if edges is not None:
                self.add_edges_from(edges)

        self.node2neigh: Dict[Any, float] = {}
        self.m: float = self.size(weight="weight")

        self.communities: Dict[Any, List[Any]] = {}
        self.sigma_tot: Dict[Any, float] = {}

        self.init_params()

    def init_params(self) -> None:
        for node in self.nodes:
            self.communities[node] = [node]
            self.nodes[node]["community"] = node
            self.node2neigh[node] = sum(
                edge_data.get("weight", 1.0)
                for _, _, edge_data in self.edges(node, data=True)
            )
            self.sigma_tot[node] = 0.0

    def update_cnt(
        self, node, old_community, new_community, neighborhood: Dict
    ) -> None:
        self.communities[old_community].remove(node)
        self.communities[new_community].append(node)
        self.nodes[node]["community"] = new_community
        self.sigma_tot[new_community] += neighborhood.get(new_community, 0.0)
        self.sigma_tot[old_community] -= neighborhood.get(old_community, 0.0)

    def get_partition(self) -> Dict:
        return self.communities.copy()

    def get_neighborhood(self, node) -> Dict:
        neighborhood = defaultdict(float)
        for neighbor in self[node]:
            c = self.nodes[neighbor]["community"]
            w = self[node][neighbor].get("weight", 1.0)
            neighborhood[c] += w

        return neighborhood

    def get_community_neighborhood(self, community) -> Dict:
        neighborhood = defaultdict(float)
        for node in community:
            for neighbor in self[node]:
                c = self.nodes[neighbor]["community"]
                w = self[node][neighbor].get("weight", 1.0)
                if neighbor in community:
                    neighborhood[c] += w / 2.0
                else:
                    neighborhood[c] += w

        return neighborhood

    def get_community_number(self) -> int:
        return sum(1 for comm in self.communities.values() if comm)

    def aggregate(self):
        return self._aggregate(self)

    @classmethod
    def _aggregate(cls, inst, communities=None):
        """Labels are discarded after aggregation. (label_returned=index only)"""

        if communities is None:
            communities = inst.communities

        else:
            assert isinstance(communities, Dict), (
                "Paramater 'communities' should be a Dict type"
            )

        G = nx.Graph()
        for node, community in inst.communities.items():
            if community:
                G.add_node(node)
                neighborhood = inst.get_community_neighborhood(community)
                for neighbor, weight in neighborhood.items():
                    G.add_edge(node, neighbor, weight=weight)

        return cls(G)

    def draw(
        self,
        fig,
        ax,
        iterations: int = 50,
        scale: float = 15.0,
        k: float = 2.0,
        edge_alpha: float = 0.25,
        node_size: float = 500.0,
        edge_width: float = 2.0,
        locally: bool = False,
        bfs_depth: int = 2,
        cmap: str = "inferno",
    ) -> None:
        G = self
        if locally:
            G = self._extract_local_subgraph(depth=bfs_depth)

        positions = nx.spring_layout(
            G, scale=scale, k=k / np.sqrt(self.order()), iterations=iterations
        )

        degrees = dict(nx.degree(G, weight="weight"))
        nws = np.array([weight for weight in degrees.values()])
        nws = nws / np.max(nws) * node_size

        ews = np.array(
            [
                data.get("weight", 1) * (v1 != v2) + 1
                for v1, v2, data in G.edges(data=True)
            ]
        )
        ews = np.log2(ews)
        ews = ews / np.max(ews)
        ews = np.clip(ews, 0.1, None) * edge_width

        attrs = nx.get_node_attributes(G, "community")
        unique = sorted(set(attrs.values()))
        idx = {val: i for i, val in enumerate(unique)}

        color_idx = [idx[attrs[n]] for n in G.nodes()]
        edge_color_idx = [(idx[attrs[u]] + idx[attrs[v]]) / 2 for u, v in G.edges()]

        cmap_obj = plt.get_cmap(cmap, len(unique))

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        nx.draw_networkx_edges(
            G,
            positions,
            ax=ax,
            edge_cmap=cmap_obj,
            edge_color=edge_color_idx,
            width=ews,
            alpha=edge_alpha,
        )

        nodes = nx.draw_networkx_nodes(
            G,
            positions,
            ax=ax,
            cmap=cmap_obj,
            node_color=color_idx,
            nodelist=degrees,
            node_size=nws,
        )

        self._animation = None

        if nodes is not None:
            nodes.set_visible(True)

            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            annot.set_visible(False)

            node_list = list(G.nodes())
            idx_to_node = {i: node for i, node in enumerate(node_list)}
            hovered_index = [None]

            def update_annot(ind):
                i = ind["ind"][0]
                node = idx_to_node[i]
                annot.xy = positions[node]
                annot.set_text(f"Node {node}\nCommunity {attrs[node]}")

            def hover(event):
                if event.inaxes == ax:
                    cont, ind = nodes.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        hovered_index[0] = ind["ind"][0]
                        fig.canvas.draw_idle()
                    else:
                        if annot.get_visible():
                            annot.set_visible(False)
                            hovered_index[0] = None
                            fig.canvas.draw_idle()

            def animate(frame):
                base_size = nws.copy()
                if hovered_index[0] is not None:
                    pulse = 1 + 0.3 * (np.sin(frame * 0.3) * 0.5 + 0.5)
                    base_size[hovered_index[0]] *= pulse
                nodes.set_sizes(base_size)
                return [nodes]

            fig.canvas.mpl_connect("motion_notify_event", hover)

            self._animation = FuncAnimation(
                fig, animate, frames=200, interval=50, blit=False, repeat=True
            )

        ax.margins(0.05)
        plt.tight_layout()

    def _extract_local_subgraph(self, depth: int) -> nx.Graph:
        nodes = list(self.nodes())[:5]
        bfs_nodes = set()
        for n in nodes:
            bfs_nodes |= set(nx.bfs_tree(self, source=n, depth_limit=depth))
        return self.subgraph(bfs_nodes)
