import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import random 

# Parameters
n = 10
target_avg_degree = 4
degree_tol = 0.2  # acceptable deviation
num_samples = 30  # per graph family

saved_graph_dir = Path('graphs/mixed')
output_dir = Path('results')
saved_graph_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)
    
def avg_degree(G):
    return sum(dict(G.degree()).values()) / G.number_of_nodes()

def accept(G, target=target_avg_degree, tol=degree_tol):
    return abs(avg_degree(G) - target) < tol

def save_edge_list_to_txt(G, filename):
    with open(filename, 'w') as file:
        for edge in G.edges():
            file.write(f"{edge[0]} {edge[1]}\n")
def main():
    # Store results
    graph_data = defaultdict(list)

    for _ in range(num_samples):
        # Erdős–Rényi
        p_er = target_avg_degree / (n - 1)
        G_er = nx.erdos_renyi_graph(n, p_er)
        graph_data["ER"].append((G_er, nx.transitivity(G_er), avg_degree(G_er)))

        # Watts–Strogatz
        G_ws = nx.watts_strogatz_graph(n, k=4, p=np.random.uniform(0, 0.5))
        graph_data["WS"].append((G_ws, nx.transitivity(G_ws), avg_degree(G_ws)))

        # Barabási–Albert
        G_ba = nx.barabasi_albert_graph(n, m=random.choice([2,3]))
        graph_data["BA"].append((G_ba, nx.transitivity(G_ba), avg_degree(G_ba)))

        # Random Regular
        try:
            G_rr = nx.random_regular_graph(d=4, n=n)
            if accept(G_rr):
                graph_data["Regular"].append((G_rr, nx.transitivity(G_rr), avg_degree(G_rr)))
        except nx.NetworkXError:
            pass  # skip infeasible regular graphs

    # Saving graphs
    graph_id_counters = defaultdict(int)  # to track per-family ID

    for family, graphs in graph_data.items():
        for G, _ in [(G, t) for G, t in zip([g[0] for g in graphs], [g[1] for g in graphs])]:
            graph_id = graph_id_counters[family]
            filename = f"{family}_{n}_{target_avg_degree}_{graph_id}.txt"
            filepath = saved_graph_dir / filename
            save_edge_list_to_txt(G, filepath)
            graph_id_counters[family] += 1

    print(f"Saved all mixed graphs to '{saved_graph_dir}'")


    # Plot transitivity distributions
    plt.figure(figsize=(10, 6))
    for name, data in graph_data.items():
        trans_vals = [t for _, t, _ in data]
        plt.hist(trans_vals, bins=10, alpha=0.6, label=name)

    plt.title(f"Transitivity of Graph Families (n={n}, avg deg ≈ {target_avg_degree})")
    plt.xlabel("Transitivity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    filepath = output_dir / Path('transitivity_count.png')
    plt.savefig(filepath, bbox_inches='tight')

    plt.figure(figsize=(5, 6))

    for name, data in graph_data.items():
        num_edges = [G.number_of_edges() for G, _, _ in data]
        trans_vals = [t for _, t, _ in data]
        plt.scatter(num_edges, trans_vals, label=name, alpha=0.7)

    plt.title(f"n={n}, avg deg ≈ {target_avg_degree}")
    plt.xlabel("Number of Edges")
    plt.ylabel("Transitivity")
    plt.legend()
    plt.tight_layout()
    filepath = output_dir / Path('transitivity_num_edges.png')
    plt.savefig(filepath, bbox_inches='tight')

    plt.figure(figsize=(5, 5))

    for name, data in graph_data.items():
        avg_degrees = [avg_degree(G) for G, _, _ in data]
        trans_vals = [t for _, t, _ in data]
        plt.scatter(avg_degrees, trans_vals, label=name, alpha=0.7)

    plt.title(f"n={n}, avg deg ≈ {target_avg_degree}")
    plt.xlabel("Average Degree")
    plt.ylabel("Transitivity")
    plt.legend()
    plt.tight_layout()
    filepath = output_dir / Path('transitivity_degree.png')
    plt.savefig(filepath, bbox_inches='tight')
    
if __name__ == '__main__':
    main()