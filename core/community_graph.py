from collections import defaultdict
from itertools import product
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class CommunityGraph(nx.Graph):
    """
    Class Implemention of Multi-Communities

    Description:
        Just a simple undirected weighted graph
    """

    def __init__(self, graph=None, vertices=None, edges=None, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)

        if graph is not None and isinstance(graph, nx.Graph):
            self.add_nodes_from(graph.nodes)
            self.add_edges_from(graph.edges.data())

        else:
            if vertices is not None:
                self.add_nodes_from(vertices)
            if edges is not None:
                self.add_edges_from(edges)

        self.node2neigh: Dict = {}
        self.m = self.size(weight="weight")

        self.communities: Dict = {}
        self.community_map: Dict = {}
        self.sigma_tot: Dict = {}

        self.init_params()

    def init_params(self) -> None:
        for node in self.nodes:
            self.communities[node] = [node]
            self.community_map[node] = node
            edges = self.edges(node, data=True)
            self.node2neigh[node] = sum(data.get("weight", 1) for _, _, data in edges)
            self.sigma_tot[node] = 0

    def update_cnt(
        self, node, old_community, new_community, neighborhood: Dict
    ) -> None:
        self.communities[old_community].remove(node)
        self.communities[new_community].append(node)
        self.community_map[node] = new_community
        self.sigma_tot[new_community] += neighborhood[new_community]
        self.sigma_tot[old_community] -= neighborhood[old_community]

    def get_partition(self) -> Dict:
        return self.communities

    def get_neighborhood(self, node) -> Dict:
        neighborhood = defaultdict(float)
        for neighbor in self[node]:
            neighbor_community = self.community_map[neighbor]
            weight = self[node][neighbor].get("weight", 1)
            neighborhood[neighbor_community] += weight  # pyright: ignore

        return neighborhood

    def get_community_neighborhood(self, community) -> Dict:
        neighborhood = defaultdict(float)
        for node in community:
            for neighbor in self[node]:
                neighbour_community = self.community_map[neighbor]
                w = self[node][neighbor].get("weight", 1)
                if neighbor in community:
                    neighborhood[neighbour_community] += w / 2  # pyright: ignore

                else:
                    neighborhood[neighbour_community] += w  # pyright: ignore

        return neighborhood

    def get_community_number(self) -> int:
        num_community = 0
        for _, community in self.communities.items():
            if community:
                num_community += 1

        return num_community

    def get_modularity(self, communities=None, resolution: float = 1.0) -> float:
        if communities is None:
            communities = self.communities

        else:
            assert isinstance(communities, Dict), (
                "Paramater 'communities' should be a Dict type"
            )

        d = dict(self.degree(weight="weight"))  # pyright: ignore
        e = self.edges

        modularity = 0

        for _, community in communities.items():
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

    def get_cpm(self, communities=None, resoluton: float = 1.0) -> float:
        if communities is None:
            communities = self.communities

        else:
            assert isinstance(communities, Dict), (
                "Paramater 'communities' should be a Dict type"
            )

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
        for index, (_, community) in enumerate(inst.communities.items()):
            if community:
                neighborhood = inst.get_community_neighborhood(community)
                for neighbor, weight in neighborhood.items():
                    G.add_edge(index, neighbor, weight=weight)

        return cls(G)

    def aggregate(self):
        return self._aggregate(self)

    def draw(
        self,
        ax,
        nodes=None,
        iterations: int = 50,
        node_size: float = 500.0,
        edge_width: float = 4.0,
        locally: bool = False,
        bfs_depth: int = 2,
        cmap: str = "viridis",
    ) -> None:
        if nodes is None and not locally:
            graph = self

        elif nodes is None and locally:
            print(
                "WARNING: Vertices to be drawn are already the entire vertices of graph, locally automatically off."
            )
            graph = self

        elif nodes is not None and locally:
            vertices = nx.compose_all(
                [
                    nx.bfs_tree(
                        self, source=node, depth_limit=bfs_depth
                    ).to_undirected()
                    for node in nodes
                ]
            ).nodes()
            graph = nx.induced_subgraph(self, vertices)

        elif nodes is not None and not locally:
            graph = nx.induced_subgraph(self, nodes)

        else:
            """Just to remove the warnings and errors"""
            graph = self

        cmap = plt.get_cmap(cmap)  # pyright: ignore
        positions = nx.spring_layout(
            graph, scale=20, k=3 / np.sqrt(self.order()), iterations=iterations
        )

        degrees = dict(graph.degree(weight="weight"))  # pyright: ignore
        indexed = [self.community_map.get(node) for node in graph]
        weights = np.array([weight for node, weight in degrees.items()])
        weights = weights / np.max(weights) * node_size

        edge_indexed = [self.community_map.get(edge[0]) for edge in graph.edges()]

        edge_weights = np.array(
            [
                data.get("weight", 1) * (v1 != v2) + 1
                for v1, v2, data in graph.edges(data=True)
            ]
        )
        edge_weights = np.log2(edge_weights)
        edge_weights = edge_weights / np.max(edge_weights)
        condition = np.logical_and(0 < edge_weights, edge_weights < 0.1)
        edge_weights[condition] = 0.1
        edge_weights = edge_weights * edge_width

        nx.draw_networkx_nodes(
            self,
            pos=positions,
            ax=ax,
            cmap=cmap,
            label=True,
            node_color=indexed,  # pyright: ignore
            nodelist=dict(graph.degree),
            node_size=weights,
        )
        nx.draw_networkx_edges(
            graph,
            ax=ax,
            edge_cmap=cmap,
            edge_color=edge_indexed,  # pyright: ignore
            width=edge_weights,
            pos=positions,
            alpha=0.2,
        )
