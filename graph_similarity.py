from utils import *


# Graphs = ['econ_wm1', 'dolphins', 'lesmis', 'polbooks', 'netscience']
Graphs = ['adjnoun', 'dolphins', 'lesmis', 'Erods', 'polbooks', 'netscience', 'USAir', 'bio_celegans']

graphs_similarity = np.zeros((len(Graphs), len(Graphs)))

matching = {}

for i, g1 in enumerate(Graphs):
    graph1 = graph_read(g1)
    for j, g2 in enumerate(Graphs):
        graph2 = graph_read(g2)
        if i >= j:
            continue
        graphs_similarity[i][j], matching[i, j] = calculate_similarity(graph1, graph2)
        graphs_similarity[j][i], matching[j, i] = calculate_similarity(graph1, graph2)

# Find the top-3 largest values
top3_largest_indices = np.argsort(-graphs_similarity, axis=1)[:, :3]


print("Top-3 Largest Indices:")
print(top3_largest_indices)

