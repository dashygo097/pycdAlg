import unittest

import algorithms.network.core as an
import networkx as nx
from utils.debuggers import timer


class TestKarate(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        G = nx.karate_club_graph()
        self.G = an.CommunityGraph(G)

    @timer
    def test_karate_louvain(self):
        G = nx.karate_club_graph()
        G = an.CommunityGraph(G)
        solver = an.LouvainSolver()
        G_agg = solver.detect(G, iterations=5, informed=True)
        print(f"Modularity : {G_agg.get_modularity()}")

    @timer
    def test_karate_louvain_cpm(self):
        solver = an.LouvainCPMSolver(resolution=0.2)
        G_agg = solver.detect(self.G, iterations=5, informed=True)
        print(f"Modularity : {G_agg.get_modularity()}")

    @timer
    def test_karate_leiden(self):
        solver = an.LeidenSolver()
        G_agg = solver.detect(self.G, depth=2, iterations=3, informed=True)
        print(f"Modularity : {G_agg.get_modularity()}")


if __name__ == "__main__":
    unittest.main()
