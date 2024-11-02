import torch.nn as nn
import torch.optim as optim
from graph_auto import *
from utils import *
import os


class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MappingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def network_alignment(g1, g2, iterations, mapping_dim, learning_rate):
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert G1 and G2 to networkx graph objects
    G1 = graph_read(g1)
    G2 = graph_read(g2)

    G1_nx = nx.Graph()
    G1_nx.add_nodes_from(range(G1.vcount()))
    G1_nx.add_edges_from(G1.get_edgelist())

    G2_nx = nx.Graph()
    G2_nx.add_nodes_from(range(G2.vcount()))
    G2_nx.add_edges_from(G2.get_edgelist())

    # Initialize mapping networks and move them to the device
    hidden_dim = 64  # Assuming embeddings have 64 dimensions
    mapping_G1_to_G2 = MappingNetwork(hidden_dim, mapping_dim).to(device)
    mapping_G2_to_G1 = MappingNetwork(hidden_dim, mapping_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(mapping_G1_to_G2.parameters()) + list(mapping_G2_to_G1.parameters()), lr=learning_rate
    )

    # Load embeddings and move them to the device
    embeddings_path1 = f'result/embeddings/{g1}_embedding.npy'
    embeddings_path2 = f'result/embeddings/{g2}_embedding.npy'

    G1_embeddings = np.load(embeddings_path1)
    G2_embeddings = np.load(embeddings_path2)

    # Convert embeddings to tensors and move them to the device
    G1_embeddings = torch.tensor(G1_embeddings, dtype=torch.float32).to(device)
    G2_embeddings = torch.tensor(G2_embeddings, dtype=torch.float32).to(device)

    # -------------------- Optimizations Begin --------------------

    # Move invariant computations outside the training loop

    # Generate network alignments based on node degree ranking
    community_G1 = find_communities(G1, 'fastgreedy')
    community_G2 = find_communities(G2, 'fastgreedy')

    node_degrees_G1 = G1.degree()
    sorted_nodes_G1 = [
        sorted(community, key=lambda x: node_degrees_G1[x], reverse=True) for community in community_G1
    ]

    node_degrees_G2 = G2.degree()
    sorted_nodes_G2 = [
        sorted(community, key=lambda x: node_degrees_G2[x], reverse=True) for community in community_G2
    ]

    _, matchings = calculate_similarity(G1, G2)

    anchored_nodes_big = []
    anchored_nodes_small = []
    for matching in matchings:
        c_1, c_2 = matching
        node_g1 = sorted_nodes_G1[c_1]
        node_g2 = sorted_nodes_G2[c_2]
        if len(node_g1) < 3 or len(node_g2) < 3:
            continue
        for i in range(3):
            node1 = node_g1[i]
            node2 = node_g2[i]
            anchored_nodes_big.append([node1, node2])

            # Find least-degree neighbors within the community
            neighbors_g1 = G1.neighbors(node1)
            neighbors_within_community_g1 = [n for n in neighbors_g1 if n in node_g1]
            sorted_neighbors_g1 = sorted(neighbors_within_community_g1, key=lambda x: G1.degree(x))
            least_degree_neighbors_g1 = sorted_neighbors_g1[:2]

            neighbors_g2 = G2.neighbors(node2)
            neighbors_within_community_g2 = [n for n in neighbors_g2 if n in node_g2]
            sorted_neighbors_g2 = sorted(neighbors_within_community_g2, key=lambda x: G2.degree(x))
            least_degree_neighbors_g2 = sorted_neighbors_g2[:2]

            combinations_small = list(itertools.product(least_degree_neighbors_g1, least_degree_neighbors_g2))
            anchored_nodes_small.extend(combinations_small)

    # Prepare batch data for supervised learning
    if anchored_nodes_big:
        inputs_big = G1_embeddings[[pair[0] for pair in anchored_nodes_big]]
        targets_big = G2_embeddings[[pair[1] for pair in anchored_nodes_big]]
    else:
        inputs_big = torch.empty(0, G1_embeddings.shape[1]).to(device)
        targets_big = torch.empty(0, G2_embeddings.shape[1]).to(device)

    if anchored_nodes_small:
        inputs_small = G1_embeddings[[pair[0] for pair in anchored_nodes_small]]
        targets_small = G2_embeddings[[pair[1] for pair in anchored_nodes_small]]
    else:
        inputs_small = torch.empty(0, G1_embeddings.shape[1]).to(device)
        targets_small = torch.empty(0, G2_embeddings.shape[1]).to(device)

    # -------------------- Optimizations End --------------------

    # Perform training iterations
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Perform unsupervised learning
        G1_to_G2 = mapping_G1_to_G2(G1_embeddings)
        G2_to_G1 = mapping_G2_to_G1(G2_embeddings)
        G1_to_G2_to_G1 = mapping_G2_to_G1(G1_to_G2)
        G2_to_G1_to_G2 = mapping_G1_to_G2(G2_to_G1)
        loss_total = 0
        # Calculate unsupervised loss
        loss_unsupervised = criterion(G1_to_G2_to_G1, G1_embeddings) + criterion(G2_to_G1_to_G2, G2_embeddings)
        loss_total += loss_unsupervised

        # Supervised learning on big anchored nodes
        if inputs_big.size(0) > 0:
            mapped_data_G1_to_G2_big = mapping_G1_to_G2(inputs_big)
            mapped_data_G2_to_G1_big = mapping_G2_to_G1(targets_big)
            loss_supervised_big = criterion(mapped_data_G1_to_G2_big, targets_big) + criterion(mapped_data_G2_to_G1_big, inputs_big)
            loss_total += loss_supervised_big
        else:
            loss_supervised_big = torch.tensor(0.0).to(device)

        # Supervised learning on small anchored nodes
        if inputs_small.size(0) > 0:
            mapped_data_G1_to_G2_small = mapping_G1_to_G2(inputs_small)
            mapped_data_G2_to_G1_small = mapping_G2_to_G1(targets_small)
            loss_supervised_small = criterion(mapped_data_G1_to_G2_small, targets_small) + criterion(mapped_data_G2_to_G1_small, inputs_small)
            loss_total += loss_supervised_small / 4  # Adjust scaling as needed
        else:
            loss_supervised_small = torch.tensor(0.0).to(device)

        loss_total.backward()
        optimizer.step()

        # Reduce logging frequency
        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch: {epoch + 1}/{iterations}, "
                f"Total Loss: {loss_total.item():.4f}, "
                f"Unsupervised Loss: {loss_unsupervised.item():.4f}, "
                f"Supervised Loss (big): {loss_supervised_big.item():.4f}, "
                f"Supervised Loss (small): {loss_supervised_small.item():.4f}"
            )

    # Generate network alignments
    with torch.no_grad():
        # Compute mapped embeddings
        mapped_embeddings_G1_to_G2 = mapping_G1_to_G2(G1_embeddings)
        mapped_embeddings_G2_to_G1 = mapping_G2_to_G1(G2_embeddings)

        # Compute distances and find alignments
        distances_G1_to_G2 = torch.cdist(mapped_embeddings_G1_to_G2, G2_embeddings)
        distances_G2_to_G1 = torch.cdist(mapped_embeddings_G2_to_G1, G1_embeddings)

        # Align G1 to G2
        _, indices = torch.min(distances_G1_to_G2, dim=1)
        align_G1_to_G2 = {i: idx.item() for i, idx in enumerate(indices)}

        # Align G2 to G1
        _, indices = torch.min(distances_G2_to_G1, dim=1)
        align_G2_to_G1 = {i: idx.item() for i, idx in enumerate(indices)}

    return align_G1_to_G2, align_G2_to_G1


# Assuming the existence of a `network_alignment` function and `save_dictionary` function

# Graphs list
Graphs = ['adjnoun', 'dolphins', 'lesmis', 'Erods', 'polbooks', 'netscience', 'USAir', 'bio_celegans']

# Load the graphs similarity matrix
graphs_similarity = np.load('result/graphs_similarity.npy')

# Find the top-3 largest values for each network
top3_largest_indices = np.argsort(graphs_similarity, axis=1)[:, -3:]

# Parameters for network alignment
iterations = 3000
mapping_dim = 64
learning_rate = 0.002

# Loop over each graph and get top-3 similar graphs for alignment
for i, g1 in enumerate(Graphs):
    print('The current network is:', g1)
    top3_indices = top3_largest_indices[i]

    for j in top3_indices:
        g2 = Graphs[j]
        print('The assisted network is:', g2)

        # Skip self-alignment
        if g1 == g2:
            continue

        # Define file paths for saving mappings
        path1 = f'result/mapping/{g1}_to_{g2}.pkl'
        path2 = f'result/mapping/{g2}_to_{g1}.pkl'

        # Check if mappings already exist, if they do, skip to the next pair
        if os.path.exists(path1) and os.path.exists(path2):
            print(f"Mapping between {g1} and {g2} already exists. Skipping...")
            continue

        # Perform network alignment if mappings do not exist
        print(f"Performing alignment between {g1} and {g2}...")
        align_G1_to_G2, align_G2_to_G1 = network_alignment(g1, g2, iterations, mapping_dim, learning_rate)

        # Save the results
        save_dictionary(align_G1_to_G2, path1)
        save_dictionary(align_G2_to_G1, path2)

        print(f"Saved mapping for {g1}_to_{g2} and {g2}_to_{g1}")