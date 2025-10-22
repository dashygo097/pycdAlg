from itertools import product
from typing import Dict, Optional

import networkx as nx

from ..community_graph import CommunityGraph


class CommunityMetrics:
    @staticmethod
    def modularity(
        graph: CommunityGraph,
        communities: Optional[Dict] = None,
        resolution: float = 1.0,
    ) -> float:
        communities = communities or graph.get_partition()
        non_empty = [c for c in communities.values() if c]
        d = dict(nx.degree(graph, weight="weight"))
        e = graph.edges
        total_weight = graph.total_weight

        modularity = 0.0

        for community in non_empty:
            for v1, v2 in product(community, repeat=2):
                try:
                    w = e[v1, v2].get("weight", 1)
                except KeyError:
                    w = 0.0

                if v1 == v2:
                    w *= 2

                modularity += w - resolution * float(d[v1]) * float(d[v2]) / (
                    2 * total_weight
                )

        return modularity / (2 * total_weight)

    @staticmethod
    def cpm(
        graph: CommunityGraph,
        communities: Optional[Dict] = None,
        resolution: float = 1.0,
    ) -> float:
        communities = communities or graph.get_partition()
        non_empty = [c for c in communities.values() if c]
        total_weight = graph.total_weight
        cpm = 0

        for community in non_empty:
            subgraph = graph.subgraph(community)
            num_edges = nx.number_of_edges(subgraph)
            num_nodes = nx.number_of_nodes(subgraph)
            cpm += num_edges - resolution * num_nodes * (num_nodes - 1) / (
                2 * total_weight
            )

        return cpm

    @staticmethod
    def community_level_seperations(
        graph: CommunityGraph, communities: Optional[Dict] = None
    ) -> int:
        communities = communities or graph.get_partition()
        non_empty = [c for c in communities.values() if c]
        community_components = 0

        for community in non_empty:
            subgraph = graph.subgraph(community)
            community_components += nx.number_connected_components(subgraph)

        return community_components - graph.community_number()
