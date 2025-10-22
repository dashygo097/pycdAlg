from typing import Any, Dict

from ...web_graph import WebGraph


class PageRank:
    def __init__(self, graph: WebGraph, damping_factor: float = 0.85) -> None:
        self.graph = graph
        self._damping_factor: float = damping_factor
        self._scores: Dict[Any, float] = {node: 1.0 for node in graph.nodes}
