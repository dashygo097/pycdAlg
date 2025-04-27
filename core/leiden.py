from collections import deque

import networkx as nx
from numpy import random
from termcolor import colored
from tqdm import tqdm

from .community_graph import CommunityGraph
from .louvain import LouvainSolver


class LeidenSolver(LouvainSolver):
    """
    Class Implementation of the Leiden Alforithm

    Description:
        The Leiden algorithm is a community detection algorithm that improves upon the Louvain method.
        It refines the communities found by Louvain and ensures that they are well-defined and non-overlapping.
        The algorithm consists of three main phases: local moving, refinement, and aggregation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = deque()
        self.v = {}

    def reset(self) -> None:
        self.queue = deque()
        self.v = {}

    def _get_name(self) -> str:
        return "Leiden"

    # NOTE: Leiden specific
    def fast_local_move(self, G: CommunityGraph) -> None:
        while self.queue:
            node = self.queue.popleft()
            self.v[node] = 0
            old_community = G.community_map[node]
            neighborhood = G.get_neighborhood(node)

            self.move_node(G, node, neighborhood)
            new_community = G.community_map[node]

            if old_community == new_community:
                continue

            for neighbor in nx.neighbors(G, node):
                if G.community_map[neighbor] == old_community and self.v[neighbor] == 0:
                    self.queue.append(neighbor)
                    self.v[neighbor] = 1
                    break

    # NOTE: Leiden specific
    def refine(self, G: CommunityGraph) -> None:
        communities = G.get_partition()
        for community in communities.values():
            if not community:
                continue

            induced_graph = nx.induced_subgraph(G, community)
            induced_graph = CommunityGraph(induced_graph)
            self.forward(induced_graph, iterations=1, is_shuffle=True, tqdm_bar=False)
            for node in community:
                self.update_refinement(G, induced_graph, node)

    def update_refinement(
        self, G: CommunityGraph, induced_graph: CommunityGraph, node
    ) -> None:
        G.communities[G.community_map[node]].remove(node)
        G.communities[induced_graph.community_map[node]].append(node)
        G.community_map[node] = induced_graph.community_map[node]

    def sync(self, G: CommunityGraph, G_reg: CommunityGraph) -> None:
        super().sync(G, G_reg)

        q_len = self.queue.__len__()

        for index in range(q_len):
            self.queue[index] = G.community_map[self.queue[index]]

        for node in G.nodes():
            self.v[node] = 1

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
            if not self.queue or not self.v:
                nodes = list(G.nodes)
                if is_shuffle:
                    random.shuffle(nodes)

                self.queue = deque(nodes)

                for node in G.nodes():
                    self.v[node] = 1

            self.fast_local_move(G)

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
        pbar = tqdm(
            total=depth + 1, desc=colored(name + " Algorithm Progress", "green")
        )
        for level in range(depth + 1):
            self.forward(G_reg, iterations, is_shuffle=is_shuffle, level=level)

            # Leiden Refinement
            pbar.set_description_str(
                colored("Refining Communities... ", "green")
                + "At "
                + colored("LEVEL", "red")
                + colored(str(level), "red")
            )
            self.refine(G_reg)

            self.sync(G, G_reg)

            pbar.set_description_str(colored("Aggregating Communities...", "green"))
            G_reg = G.aggregate()

            pbar.set_description_str(colored(name + " Algorithm Progress", "green"))
            pbar.update(1)

        pbar.close()

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
