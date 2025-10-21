import unittest

import matplotlib.pyplot as plt
import networkx as nx

import pycd


class TestKarateDraw(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_karate_louvain_draw(self):
        print("--- Louvain ---")
        G = nx.karate_club_graph()
        G = pycd.CommunityGraph(G)
        solver = pycd.LouvainSolver(resolution=1.0)
        G_ = solver.detect(G, iterations=5, informed=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        G.draw(fig, ax)
        plt.show()
        print(f"Modularity : {pycd.CommunityMetrics.modularity(G_)}")


if __name__ == "__main__":
    unittest.main()
