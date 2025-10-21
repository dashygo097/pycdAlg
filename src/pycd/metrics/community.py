from ..community_graph import CommunityGraph
import networkx as nx
from typing import Dict, Optional
from itertools import product


class CommunityMetrics:
    @staticmethod
    def modularity(
        graph: CommunityGraph,
        communities: Optional[Dict] = None,
        resolution: float = 1.0,
    ) -> float:
        communities = communities or graph.communities
        d = dict(graph.degree(weight="weight"))
        e = graph.edges
        m = graph.m

        modularity = 0

        for community in communities.values():
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
        m = graph.m
        cpm = 0

        for community in communities.values():
            subG = nx.induced_subgraph(graph, community)
            e_c = nx.number_of_edges(subG)
            n_c = nx.number_of_nodes(subG)
            cpm += e_c - resolution * n_c * (n_c - 1) / (2 * m)

        return cpm
