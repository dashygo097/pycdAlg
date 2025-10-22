from typing import Any, List, Optional

import networkx as nx
from matplotlib.axis import Axis
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
