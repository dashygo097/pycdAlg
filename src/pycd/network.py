from abc import ABC, abstractmethod
import networkx as nx


class Network(ABC, nx.Graph): ...
