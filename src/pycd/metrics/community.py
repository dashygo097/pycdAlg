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
        communities = communities or graph.communities
        non_empty = [c for c in communities.values() if c]
        d = dict(nx.degree(graph, weight="weight"))
        e = graph.edges
        m = graph.m

        modularity = 0

        for community in non_empty:
            for v1, v2 in product(community, repeat=2):
                try:
                    w = e[v1, v2].get("weight", 1)
                except KeyError:
                    w = 0

                if v1 == v2:
                    w *= 2

                modularity += w - resolution * float(d[v1]) * float(d[v2]) / (2 * m)

        return modularity / (2 * m)

    @staticmethod
    def cpm(
        graph: CommunityGraph,
        communities: Optional[Dict] = None,
        resolution: float = 1.0,
    ) -> float:
        communities = communities or graph.communities
        non_empty = [c for c in communities.values() if c]
        m = graph.m
        cpm = 0

        for community in non_empty:
            subgraph = graph.subgraph(community)
            e_c = nx.number_of_edges(subgraph)
            n_c = nx.number_of_nodes(subgraph)
            cpm += e_c - resolution * n_c * (n_c - 1) / (2 * m)

        return cpm

    @staticmethod
    def community_level_seperations(
        graph: CommunityGraph, communities: Optional[Dict] = None
    ) -> int:
        communities = communities or graph.communities
        non_empty = [c for c in communities.values() if c]
        community_components = 0

        for community in non_empty:
            subgraph = graph.subgraph(community)
            community_components += nx.number_connected_components(subgraph)

        return community_components - graph.get_community_number()
