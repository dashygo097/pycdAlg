import unittest

import networkx as nx
from utils.debuggers import timer

import pycd


class TestWebND(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        G = nx.read_edgelist(
            "./assets/web-NotreDame.txt",
            create_using=nx.Graph(),
            nodetype=int,
            data=[("weight", float)],  # pyright: ignore
        )
        self.G = pycd.CommunityGraph(G)

    @timer
    def test_webND_louvain(self):
        solver = pycd.LouvainSolver()
        G_agg = solver.detect(self.G, depth=2, iterations=2, informed=True)
        print(f"Modularity: {G_agg.get_modularity()}")
        print("")

    @timer
    def test_webND_leiden(self):
        solver = pycd.LeidenSolver()
        G_agg = solver.detect(self.G, depth=2, iterations=2, informed=True)
        print(f"Modularity: {G_agg.get_modularity()}")
        print("")


if __name__ == "__main__":
    unittest.main()
