import unittest

import networkx as nx

import pycd


class TestMultiKarate(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    @pycd.timer
    def test_karate_louvain_for_many_times(self):
        solver = pycd.LouvainSolver()
        modularity = []
        for _ in range(1000):
            G = nx.karate_club_graph()
            G = pycd.CommunityGraph(G)
            G_ = solver.detect(G, iterations=7, informed=False)
            modularity.append(G_.modularity())
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
            modularity.append(G_.modularity())

        print(f"Average modularity : {sum(modularity) / len(modularity)}")
        print(
            f"Varipycdce : {sum((x - sum(modularity) / len(modularity)) ** 2 for x in modularity) / len(modularity)}"
        )
        print("")


if __name__ == "__main__":
    unittest.main()
