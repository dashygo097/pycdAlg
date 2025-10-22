from typing import Any, List, Optional

import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .network import Network


class WebGraph(Network):
    """
    A class to represent a web graph, inheriting from Network.
    This class is designed
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        vertices: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        drop_unconnected: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(graph, vertices, edges, drop_unconnected, **kwargs)

    def initialize(self) -> None:
        self.adjacency_matrix = nx.adjacency_matrix(self).to_numpy_array()

    def draw(self, fig: Figure, ax: Axes, **kwargs) -> None: ...
