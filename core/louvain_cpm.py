from typing import Dict

from .community_graph import CommunityGraph
from .louvain import LouvainSolver


class LouvainCPMSolver(LouvainSolver):
    """
    Class Implementation of the Louvain algorithm with the Constant Potts Model (CPM) method.

    Description:
        CPM is a method for community detection in networks that uses a Potts model to optimize the modularity of the network.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_name(self):
        return "LouvainCPM"

    # NOTE: CPM specific
    def move_node(self, G: CommunityGraph, node, neighborhood: Dict) -> None:
        delta_Q = 0
        delta_C = -1
        for community in neighborhood.keys():
            ki_in = neighborhood[community]
            nc_new = len(G.communities[community])
            nc_old = len(G.communities[G.community_map[node]])

            delta = ki_in - self.resolution * (nc_new - nc_old + 1)

            if delta_Q < delta:
                delta_Q = delta
                delta_C = community

        if delta_C >= 0:
            G.update_cnt(node, G.community_map[node], delta_C, neighborhood)
