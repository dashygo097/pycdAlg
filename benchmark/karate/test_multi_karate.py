import unittest

import algorithms.network.core as an
import networkx as nx
from utils.debuggers import timer


class TestMultiKarate(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    @timer
    def test_karate_louvain_for_many_times(self):
        solver = an.LouvainSolver()
        modularity = []
        for _ in range(1000):
            G = nx.karate_club_graph()
            G = an.CommunityGraph(G)
            G_agg = solver.detect(G, iterations=5, informed=False)
            modularity.append(G_agg.get_modularity())
        print(f"Average modularity : {sum(modularity) / len(modularity)}")
        print(
            f"Variance : {sum((x - sum(modularity) / len(modularity)) ** 2 for x in modularity) / len(modularity)}"
        )
        print("")

    @timer
    def test_karate_leiden_for_many_times(self):
        solver = an.LeidenSolver()
        modularity = []
        for _ in range(1000):
            G = nx.karate_club_graph()
            G = an.CommunityGraph(G)
            G_agg = solver.detect(G, depth=1, iterations=2, informed=False)
            modularity.append(G_agg.get_modularity())

        print(f"Average modularity : {sum(modularity) / len(modularity)}")
        print(
            f"Variance : {sum((x - sum(modularity) / len(modularity)) ** 2 for x in modularity) / len(modularity)}"
        )
        print("")


if __name__ == "__main__":
    unittest.main()
