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

    def __init__(
        self,
        resolution: float = 1.0,
        alpha: float = 0.9,
        beta: float = 2.0,
        allow_negative_move: bool = True,
        negative_move_prob: float = 0.1,
        negative_move_weight: float = 0.1,
    ) -> None:
        self.resolution = resolution
        self.alpha = alpha
        self.beta = beta

        self.beta_runtime = beta

        # Parameters for negative move
        self.allow_negative_move = allow_negative_move
        self.negative_move_prob = negative_move_prob
        self.negative_move_weight = negative_move_weight

    def reset(self) -> None:
        pass

    def _get_name(self) -> str:
        return "Louvain"

    def beta_schedule(self, total: int, epoch: int) -> None:
        if epoch < total // 2:
            self.beta_runtime = self.beta
        else:
            self.beta_runtime = self.beta_runtime * self.alpha

    def move_node(self, G: CommunityGraph, node, neighborhood: Dict) -> bool:
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
            # NOTE: delta_C turns out to be np.int64, not int
            # NOTE: Maybe this is an issue with numpy's version
            delta_C = int(delta_C) if isinstance(delta_C, np.integer) else delta_C

        if delta_C > -1:
            G.update_cnt(node, G.community_map[node], delta_C, neighborhood)
            return True

        else:
            return False

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
            self.beta_schedule(iterations, iteration)
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

        if informed:
            pbar = tqdm(
                range(depth + 1),
                desc=colored(name + " Algorithm Progress", "green"),
            )
        else:
            pbar = range(depth + 1)

        for level in pbar:
            self.forward(
                G_reg,
                iterations,
                is_shuffle=is_shuffle,
                level=level,
                tqdm_bar=True if informed else False,
            )
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
