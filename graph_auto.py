import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE  # GAE已经内置在torch_geometric中
from torch_geometric.nn import GCNConv  # torch_geometric内置GCN层
import networkx as nx
from torch_geometric.utils import from_networkx, negative_sampling
from utils import *


class GCN_Encoder(torch.nn.Module):  # 两层GCN构成的Encoder
    def __init__(self, in_channels, out_channels):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)


# Create a random graph using networkx
def autoencouder(G):
    graph = nx.Graph()
    graph.add_nodes_from(range(G.vcount()))
    graph.add_edges_from(G.get_edgelist())

    degree_centrality = G.degree()

    # Compute closeness centrality
    closeness_centrality = G.closeness()

    # Compute clustering coefficient
    community = find_communities(G, 'fastgreedy').membership

    num_features = len(set(community)) + 2
    # Convert the graph to torch_geometric format
    data = from_networkx(graph)
    data.x = torch.zeros(data.num_nodes, num_features)

    # Assign the one-hot encoded community labels as node features
    for i, com in enumerate(community):
        data.x[i, com] = 1
    data.x[:, -1] = torch.tensor(degree_centrality)
    data.x[:, -2] = torch.tensor(closeness_centrality)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Set model parameters
    in_channels = num_features  # Number of input features
    out_channels = 64  # Number of output features
    epochs = 5000  # Number of training epochs

    # Build the model
    encoder = GCN_Encoder(in_channels, out_channels).to(device)
    model = GAE(encoder).to(device)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        # Positive edges for training
        pos_edge_index = data.edge_index

        # Negative edges for training (using negative sampling)
        neg_edge_index = negative_sampling(pos_edge_index, G.vcount()).to(device)

        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    z = model.encode(data.x, data.edge_index)
    return z