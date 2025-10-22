from typing import Dict

import numpy as np
import numpy.random as random

from ...community_graph import CommunityGraph
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
    def move_node(self, graph: CommunityGraph, node, neighborhood: Dict) -> bool:
        delta_C = None
        communities = []
        weights = []

        nc_old = len(graph.communities[graph.nodes[node]["community"]])

        for community in neighborhood.keys():
            ki_in = neighborhood[community]
            nc_new = len(graph.communities[community])

            delta = ki_in - self.resolution * (nc_new - nc_old + 1)

            if delta > 0:
                weights.append(delta * self.beta_runtime)
                communities.append(community)

            elif (
                self.allow_negative_move
                and np.random.random() < self.negative_move_prob
            ):
                weights.append(self.negative_move_weight * self.beta_runtime)
                communities.append(community)

        if any(w > 0 for w in weights):
            w_max = np.max(weights)
            weights = np.exp(weights - w_max)
            weights /= np.sum(weights, axis=0)
            delta_C = random.choice(communities, p=weights)
            # NOTE: delta_C turns out to be numpy typed
            # NOTE: Maybe this is an issue with numpy's version
            delta_C = delta_C.astype(object) if delta_C is not None else None

        if delta_C is not None and delta_C != graph.nodes[node]["community"]:
            graph.update_cnt(
                node, graph.nodes[node]["community"], delta_C, neighborhood
            )
            return True

        else:
            return False
