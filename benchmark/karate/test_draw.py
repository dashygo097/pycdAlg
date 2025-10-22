import unittest

import matplotlib.pyplot as plt
import networkx as nx

import pycd


class TestKarateDraw(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_karate_louvain_draw(self):
        print("--- Louvain ---")
        graph = nx.karate_club_graph()
        graph = pycd.CommunityGraph(graph)
        solver = pycd.LouvainSolver(resolution=1.0)
        graph_ = solver.detect(graph, iterations=5, informed=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        graph.draw(fig, ax)
        plt.show()
        print(f"Modularity : {pycd.CommunityMetrics.modularity(graph_)}")


if __name__ == "__main__":
    unittest.main()
