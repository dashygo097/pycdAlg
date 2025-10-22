import cProfile
import io
import pstats

import networkx as nx

import pycd

if __name__ == "__main__":
    solver = pycd.LeidenSolver()

    pr = cProfile.Profile()
    pr.enable()

    modularity = []
    for _ in range(1000):
        graph = nx.karate_club_graph()
        graph = pycd.CommunityGraph(graph)
        graph_ = solver.detect(graph, depth=1, iterations=5, informed=False)
        modularity.append(pycd.CommunityMetrics.modularity(graph_))

    print(f"Average modularity : {sum(modularity) / len(modularity)}")
    print(
        f"Varipycdce : {sum((x - sum(modularity) / len(modularity)) ** 2 for x in modularity) / len(modularity)}"
    )
    print("")

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # Show top 20 functions
    print(s.getvalue())
