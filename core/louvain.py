from typing import Dict

import numpy as np
import numpy.random as random
from termcolor import colored
from tqdm import tqdm

from .community_graph import CommunityGraph


class LouvainSolver:
    """
    Class Implemetation of Louvain Algorithm

    Description:
        The Louvain method is a popular algorithm for community detection in networks.
        It optimizes modularity by iteratively grouping nodes into communities based on their connections.
        The algorithm consists of two main phases: local optimization and aggregation.
    """

    def __init__(self, resolution: float = 1.0, beta: float = 2.0) -> None:
        self.resolution = resolution
        self.beta = beta

    def reset(self) -> None:
        pass

    def _get_name(self) -> str:
        return "Louvain"

    def move_node(self, G: CommunityGraph, node, neighborhood: Dict) -> None:
        delta_C = -1
        communities = []
        weights = []
        # random choose a community but with more delta, the more likely to be chosen
        for community in neighborhood.keys():
            ki_in = neighborhood[community]
            ki = G.node2neigh[node]
            tot = G.sigma_tot[community]

            delta = ki_in - self.resolution * ki * tot / (2 * G.m)
            if delta > 0:
                weights.append(delta * self.beta)
                communities.append(community)

        if weights:
            w_max = np.max(weights)
            weights = np.exp(weights - w_max)
            weights /= np.sum(weights, axis=0)
            delta_C = random.choice(communities, p=weights)
            # NOTE: delta_C turns out to be np.int64, not int
            # NOTE: Maybe this is an issue with numpy's version
            delta_C = int(delta_C) if isinstance(delta_C, np.integer) else delta_C

        if delta_C > -1:
            G.update_cnt(node, G.community_map[node], delta_C, neighborhood)

    def sync(self, G: CommunityGraph, G_reg: CommunityGraph) -> None:
        for node in G.nodes:
            old_community = G.community_map[node]
            new_community = G_reg.community_map[old_community]
            if old_community != new_community:
                neighborhood = G.get_neighborhood(node)
                G.update_cnt(node, old_community, new_community, neighborhood)

    def forward(
        self,
        G: CommunityGraph,
        iterations: int = 1,
        is_shuffle: bool = True,
        level: int = 0,
        tqdm_bar: bool = True,
    ) -> None:
        for iteration in (
            tqdm(
                range(iterations),
                total=iterations,
                leave=False,
                desc="At "
                + colored("LEVEL", "red")
                + colored(str(level), "red")
                + " with "
                + colored(str(G.get_community_number()), "yellow", attrs=["bold"])
                + " vertices",
            )
            if tqdm_bar
            else range(iterations)
        ):
            nodes = list(G.nodes)
            if is_shuffle:
                random.shuffle(nodes)

            for node in nodes:
                neighborhood = G.get_neighborhood(node)
                self.move_node(G, node, neighborhood)

    def detect(
        self,
        G: CommunityGraph,
        depth: int = 0,
        iterations: int = 2,
        is_shuffle: bool = True,
        informed: bool = False,
    ) -> CommunityGraph:
        name = self._get_name()
        G_reg = G
        for level in tqdm(
            range(depth + 1),
            desc=colored(name + " Algorithm Progress", "green"),
        ):
            self.forward(G_reg, iterations, is_shuffle=is_shuffle, level=level)
            self.sync(G, G_reg)

            G_reg = G.aggregate()

        if informed:
            print("done!")
            print(
                "Current State: "
                + colored(f"LEVEL{depth}", "red", attrs=["bold"])
                + " with "
                + colored(f"{G_reg.get_community_number()}", "yellow", attrs=["bold"])
                + " communities"
            )
        self.reset()
        return G_reg
