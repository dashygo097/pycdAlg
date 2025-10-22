from collections import deque
from typing import Any, Dict

from numpy import random
from termcolor import colored
from tqdm import tqdm

from ...community_graph import CommunityGraph
from .louvain import LouvainSolver


class LeidenSolver(LouvainSolver):
    """
    Class Implementation of the Leiden Alforithm

    Description:
        The Leiden algorithm is a community detection algorithm that improves upon the Louvain method.
        It refines the communities found by Louvain and ensures that they are well-defined and non-overlapping.
        The algorithm consists of three main phases: local moving, refinement, and aggregation.
    """

    def __init__(self, refine_iterations: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.refine_iterations: int = refine_iterations
        self._queue: deque = deque()
        self._v: Dict[Any, bool] = {}

    def reset(self) -> None:
        self._queue.clear()
        self._v = {}

    def _get_name(self) -> str:
        return "Leiden"

    # NOTE: Leiden specific
    def fast_local_move(self, graph: CommunityGraph) -> None:
        while self._queue:
            node = self._queue.popleft()
            if not self._v[node]:
                continue

            self._v[node] = False
            old_community = graph.nodes[node]["community"]
            neighborhood = graph.neighborhood(node)

            if not self.move_node(graph, node, neighborhood):
                continue

            new_community = graph.nodes[node]["community"]

            if old_community == new_community:
                continue

            for neighbor in graph[node]:
                if (
                    graph.nodes[neighbor]["community"] == old_community
                    and self._v[neighbor] == 0
                ):
                    self._queue.append(neighbor)
                    self._v[neighbor] = True

    # NOTE: Leiden specific
    def refine(self, graph: CommunityGraph) -> None:
        communities = graph.get_partition()
        non_empty = [c for c in communities.values() if c]

        for community in non_empty:
            induced_graph = graph.subgraph(community)
            induced_graph = CommunityGraph(induced_graph)

            self._refine_graph(induced_graph)
            self.update_refinement(graph, induced_graph)

    # NOTE: Leiden specific
    def update_refinement(
        self, graph: CommunityGraph, induced_graph: CommunityGraph
    ) -> None:
        for node in induced_graph.nodes():
            graph.update_cnt(
                node,
                graph.nodes[node]["community"],
                induced_graph.nodes[node]["community"],
                induced_graph.neighborhood(node),
            )

    def sync(self, graph: CommunityGraph, graph_: CommunityGraph) -> None:
        super().sync(graph, graph_)

        q_len = self._queue.__len__()
        for index in range(q_len):
            self._queue[index] = graph.nodes[self._queue[index]]["community"]

        self._v = {node: True for node in graph.nodes()}

    def forward(
        self,
        graph: CommunityGraph,
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
                + colored(str(graph.community_number()), "yellow", attrs=["bold"])
                + " vertices",
            )
            if tqdm_bar
            else range(iterations)
        ):
            self.beta_schedule(iterations, iteration)
            if not self._queue or not self._v:
                nodes = list(graph.nodes)
                if is_shuffle:
                    random.shuffle(nodes)

                self._queue = deque(nodes)

                for node in graph.nodes():
                    self._v[node] = True

            self.fast_local_move(graph)

    def detect(
        self,
        graph: CommunityGraph,
        depth: int = 0,
        iterations: int = 2,
        is_shuffle: bool = True,
        informed: bool = False,
    ) -> CommunityGraph:
        name = self._get_name()
        graph_ = CommunityGraph(graph)

        pbar = None
        if informed:
            pbar = tqdm(
                total=depth + 1, desc=colored(name + " Algorithm Progress", "green")
            )
        for level in range(depth + 1):
            self.forward(
                graph_,
                iterations,
                is_shuffle=is_shuffle,
                level=level,
                tqdm_bar=True if informed else False,
            )

            if informed and pbar is not None:
                # Leiden Refinement
                pbar.set_description_str(
                    colored("Refining Communities... ", "green")
                    + "At "
                    + colored("LEVEL", "red")
                    + colored(str(level), "red")
                )

            self.sync(graph, graph_)

            # Refinement
            for _ in range(self.refine_iterations):
                self.refine(graph_)

            if informed and pbar is not None:
                pbar.set_description_str(
                    colored("Syncing Communities...", "green")
                    + "At "
                    + colored("LEVEL", "red")
                    + colored(str(level), "red")
                )
                pbar.set_description_str(colored("Aggregating Communities...", "green"))
            graph_ = graph.aggregate_into()

            if informed and pbar is not None:
                pbar.set_description_str(colored(name + " Algorithm Progress", "green"))
                pbar.update(1)

        if informed and pbar is not None:
            pbar.close()

        if informed:
            print("done!")
            print(
                "Current State: "
                + colored(f"LEVEL{depth}", "red", attrs=["bold"])
                + " with "
                + colored(f"{graph_.community_number()}", "yellow", attrs=["bold"])
                + " communities"
            )
        self.reset()
        return graph_

    def _refine_graph(self, graph: CommunityGraph) -> None:
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        self._queue = deque(nodes)
        self._v = {n: True for n in nodes}

        for _ in range(self.refine_iterations):
            self.fast_local_move(graph)
