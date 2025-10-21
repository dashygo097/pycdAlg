import unittest
from pathlib import Path

import networkx as nx

import pycd


class TestWebND(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        script_dir = Path(__file__).parent

        extracted_path = script_dir / Path("../datasets/webND/web-NotreDame.txt")
        if extracted_path.exists():
            pass
        else:
            raise FileNotFoundError(f"Dataset not found: {extracted_path}")

        G = nx.read_edgelist(
            str(extracted_path),
            create_using=nx.Graph(),
            nodetype=int,
            data=[("weight", float)],  # pyright: ignore
        )

        self.G = pycd.CommunityGraph(G)

    @pycd.timer
    def test_webND_louvain(self):
        solver = pycd.LouvainSolver()
        G_ = solver.detect(self.G, depth=2, iterations=5, informed=True)
        print(f"Modularity : {pycd.CommunityMetrics.modularity(G_)}")
        print("")

    @pycd.timer
    def test_webND_leiden(self):
        solver = pycd.LeidenSolver()
        G_ = solver.detect(self.G, depth=2, iterations=5, informed=True)
        print(f"Modularity : {pycd.CommunityMetrics.modularity(G_)}")
        print("")


if __name__ == "__main__":
    unittest.main()
