from abc import ABC, abstractmethod
from typing import Any, List, Optional

import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Network(ABC, nx.Graph):
    """
    Abstract base class for specialized network implementations.

    This class extends NetworkX Graph to provide a common interface for
    various network types (community detection, flow networks, etc.).

    All subclasses must implement the abstract methods.
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        vertices: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        drop_unconnected: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._drop_unconnected = drop_unconnected

        self._build_from_input(graph, vertices, edges)
        self.initialize()

    def _build_from_input(
        self,
        base_graph: Optional[nx.Graph],
        vertices: Optional[List[Any]],
        edges: Optional[List[Any]],
    ) -> None:
        if base_graph is not None and isinstance(base_graph, nx.Graph):
            graph_to_copy = self._prepare_base_graph(base_graph)
            self.add_nodes_from(graph_to_copy.nodes(data=True))
            self.add_edges_from(graph_to_copy.edges(data=True))
        else:
            if vertices is not None:
                self.add_nodes_from(vertices)
            if edges is not None:
                self.add_edges_from(edges)

    def _prepare_base_graph(self, graph: nx.Graph) -> nx.Graph:
        if not self._drop_unconnected:
            return graph

        graph = graph.copy()
        isolated_nodes = [node for node, degree in nx.degree(graph) if degree == 0]
        if isolated_nodes:
            graph.remove_nodes_from(isolated_nodes)
        return graph

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def draw(self, fig: Figure, ax: Axes, **kwargs) -> None:
        pass
