import unittest
import networkx as nx
import matplotlib.pyplot as plt

import pycd


class TestKarate(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        G = nx.karate_club_graph()
        self.G = pycd.CommunityGraph(G)

    def test_karate_louvain_draw(self):
        G = nx.karate_club_graph()
        G = pycd.CommunityGraph(G)
        solver = pycd.LouvainSolver(resolution=1.0)
        G_ = solver.detect(G, iterations=5, informed=True)
        print(f"Modularity : {pycd.CommunityMetrics.modularity(G_)}")

    @pycd.timer
    def test_karate_louvain(self):
        G = nx.karate_club_graph()
        G = pycd.CommunityGraph(G)
        solver = pycd.LouvainSolver(resolution=1.0)
        G_ = solver.detect(G, iterations=5, informed=True)
        print(f"Modularity : {pycd.CommunityMetrics.modularity(G_)}")

    @pycd.timer
    def test_karate_louvain_cpm(self):
        solver = pycd.LouvainCPMSolver(resolution=0.2)
        G_ = solver.detect(self.G, iterations=5, informed=True)
        print(f"CPM : {pycd.CommunityMetrics.cpm(G_)}")

    @pycd.timer
    def test_karate_leiden(self):
        solver = pycd.LeidenSolver(resolution=1.0)
        G_ = solver.detect(self.G, depth=2, iterations=2, informed=True)
        print(f"Modularity : {pycd.CommunityMetrics.modularity(G_)}")

    @pycd.timer
    def test_karate_louvain_for_many_times(self):
        solver = pycd.LouvainSolver()
        modularity = []
        for _ in range(1000):
            G = nx.karate_club_graph()
            G = pycd.CommunityGraph(G)
            G_ = solver.detect(G, iterations=7, informed=False)
            modularity.append(pycd.CommunityMetrics.modularity(G_))
        print(f"Average modularity : {sum(modularity) / len(modularity)}")
        print(
            f"Varipycdce : {sum((x - sum(modularity) / len(modularity)) ** 2 for x in modularity) / len(modularity)}"
        )
        print("")

    @pycd.timer
    def test_karate_leiden_for_mpycdy_times(self):
        solver = pycd.LeidenSolver()
        modularity = []
        for _ in range(1000):
            G = nx.karate_club_graph()
            G = pycd.CommunityGraph(G)
            G_ = solver.detect(G, depth=1, iterations=3, informed=False)
            modularity.append(pycd.CommunityMetrics.modularity(G_))

        print(f"Average modularity : {sum(modularity) / len(modularity)}")
        print(
            f"Varipycdce : {sum((x - sum(modularity) / len(modularity)) ** 2 for x in modularity) / len(modularity)}"
        )
        print("")


if __name__ == "__main__":
    unittest.main()
