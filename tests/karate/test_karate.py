import unittest

import algorithms.network.core as an
import matplotlib.pyplot as plt
import networkx as nx
from utils.debuggers import timer


class TestKarate(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        G = nx.karate_club_graph()
        self.G = an.CommunityGraph(G)

    @timer()
    def test_karate_louvain(self):
        solver = an.LouvainSolver()
        G_agg = solver.detect(self.G, iterations=5, informed=True)
        print(f"Modularity : {G_agg.get_modularity()}")
        print("")

    @timer()
    def test_karate_louvain_cpm(self):
        solver = an.LouvainCPMSolver(resolution=0.2)
        G_agg = solver.detect(self.G, iterations=5, informed=True)
        print(f"Modularity : {G_agg.get_modularity()}")
        print("")

    @timer()
    def test_karate_leiden(self):
        solver = an.LeidenSolver()
        G_agg = solver.detect(self.G, depth=3, iterations=2, informed=True)
        print(f"Modularity : {G_agg.get_modularity()}")
        print("")


if __name__ == "__main__":
    unittest.main()
