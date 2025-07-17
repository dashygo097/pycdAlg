from collections import defaultdict
from itertools import product
from typing import Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class CommunityGraph(nx.Graph):
    """
    Class Implemention of Multi-Communities

    Description:
        Just a simple undirected weighted graph
    """

    def __init__(
        self, base_graph=None, vertices=None, edges=None, *arg, **kwargs
    ) -> None:
        super().__init__(*arg, **kwargs)

        if base_graph is not None and isinstance(base_graph, nx.Graph):
            self.add_nodes_from(base_graph.nodes)
            self.add_edges_from(base_graph.edges.data())
        else:
            if vertices is not None:
                self.add_nodes_from(vertices)
            if edges is not None:
                self.add_edges_from(edges)

        self.node2neigh: Dict = {}
        self.m = self.size(weight="weight")

        self.communities: Dict = {}
        self.sigma_tot: Dict = {}

        self.init_params()

    def init_params(self) -> None:
        self.modularity = 0.0

        for node in self.nodes:
            self.communities[node] = [node]
            self.nodes[node]["community"] = node
            self.node2neigh[node] = sum(
                data.get("weight", 1) for _, _, data in self.edges(node, data=True)
            )
            self.sigma_tot[node] = 0

    def update_cnt(
        self, node, old_community, new_community, neighborhood: Dict
    ) -> None:
        self.communities[old_community].remove(node)
        self.communities[new_community].append(node)
        self.nodes[node]["community"] = new_community
        self.sigma_tot[new_community] += neighborhood[new_community]
        self.sigma_tot[old_community] -= neighborhood[old_community]

    def get_partition(self) -> Dict:
        return self.communities

    def get_neighborhood(self, node) -> Dict:
        neighborhood = defaultdict(float)
        for neighbor in self[node]:
            c = self.nodes[neighbor]["community"]
            w = self[node][neighbor].get("weight", 1)
            neighborhood[c] += w

        return neighborhood

    def get_community_neighborhood(self, community) -> Dict:
        neighborhood = defaultdict(float)
        for node in community:
            for neighbor in self[node]:
                c = self.nodes[neighbor]["community"]
                w = self[node][neighbor].get("weight", 1)
                if neighbor in community:
                    neighborhood[c] += w / 2.0
                else:
                    neighborhood[c] += w

        return neighborhood

    def get_community_number(self) -> int:
        return sum(1 for comm in self.communities.values() if comm)

    def get_modularity(
        self, communities: Optional[Dict] = None, resolution: float = 1.0
    ) -> float:
        communities = communities or self.communities

        d = dict(self.degree(weight="weight"))
        e = self.edges

        modularity = 0

        for community in communities.values():
            for v1, v2 in product(community, repeat=2):
                try:
                    w = e[v1, v2].get("weight", 1)
                except KeyError:
                    w = 0

                if v1 == v2:
                    w *= 2

                modularity += w - resolution * float(d[v1]) * float(d[v2]) / (
                    2 * self.m
                )

        return modularity / (2 * self.m)

    def get_cpm(
        self, communities: Optional[Dict] = None, resoluton: float = 1.0
    ) -> float:
        communities = communities or self.communities
        cpm = 0

        for _, community in communities.items():
            subG = nx.induced_subgraph(self, community)
            e_c = nx.number_of_edges(subG)
            n_c = nx.number_of_nodes(subG)
            cpm += e_c - resoluton * n_c * (n_c - 1) / (2 * self.m)

        return cpm

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
        for index, (node, community) in enumerate(inst.communities.items()):
            if community:
                G.add_node(node)
                neighborhood = inst.get_community_neighborhood(community)
                for neighbor, weight in neighborhood.items():
                    G.add_edge(node, neighbor, weight=weight)

        return cls(G)

    def aggregate(self):
        return self._aggregate(self)

    def draw(
        self,
        ax,
        iterations: int = 50,
        node_size: float = 500.0,
        edge_width: float = 2.0,
        legend: bool = True,
        locally: bool = False,
        bfs_depth: int = 2,
        cmap: str = "viridis",
    ) -> None:
        G = self
        if locally:
            G = self._extract_local_subgraph(depth=bfs_depth)

        cmap = plt.get_cmap(cmap)
        positions = nx.spring_layout(
            G, scale=20, k=3 / np.sqrt(self.order()), iterations=iterations
        )

        degrees = dict(G.degree(weight="weight"))
        for n, d in degrees.items():
            if d == 0:
                degrees[n] = 0.5
        weights = np.array([weight for weight in degrees.values()])
        weights = weights / np.max(weights) * node_size

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
        edge_color_idx = [idx[attrs[v1]] for v1, _ in G.edges()]

        cmap_obj = plt.get_cmap(cmap, len(unique))

        nx.draw_networkx_nodes(
            G,
            pos=positions,
            ax=ax,
            label=True,
            cmap=cmap_obj,
            node_color=color_idx,
            nodelist=degrees,
            node_size=weights,
        )
        nx.draw_networkx_edges(
            G,
            ax=ax,
            edge_cmap=cmap,
            edge_color=edge_color_idx,
            width=ews,
            pos=positions,
            alpha=0.2,
        )

        ax.margins(0.05)
        plt.tight_layout()

    def _extract_local_subgraph(self, depth=2):
        nodes = list(self.nodes())[:5]
        bfs_nodes = set()
        for n in nodes:
            bfs_nodes |= set(nx.bfs_tree(self, source=n, depth_limit=depth))
        return self.subgraph(bfs_nodes)
