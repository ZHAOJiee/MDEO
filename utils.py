import itertools
import pickle
import math
import random
from sklearn import metrics
import numpy as np
import copy
from igraph import *



# Save dictionary to a file
def save_dictionary(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)


# Read dictionary from a file
def read_dictionary(file_path):
    with open(file_path, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary


def get_adjacency(graph):
    adjacency = graph.get_adjacency()
    A = np.zeros((graph.vcount(), graph.vcount()))  # 转为numpy形式
    for i in range(graph.vcount()):
        A[i][:] = adjacency[i][:]
    return A

def cross(father, mother):
    choromosome_length = len(father) - 1
    ratio_father = father[-1]
    ratio_mother = mother[-1]
    Min = min(ratio_father, ratio_mother)
    Max = max(ratio_father, ratio_mother)
    if ratio_father == Max:
        child1 = copy.deepcopy(father)
        child2 = copy.deepcopy(mother)
    else:
        child1 = copy.deepcopy(mother)
        child2 = copy.deepcopy(father)
    # cross_point = random.sample(range(0, self.choromosome_length), self.cross_num)
    cross_point = []
    cro = np.random.rand(choromosome_length).tolist()
    for i in range(choromosome_length):
        if cro[i] > 0.5:
            cross_point.append(i)

    # cross_point.sort()

    tem = []
    for i in cross_point:
        temp = child1[i]
        child1[i] = child2[i]
        child2[i] = temp
        if Min <= i < Max:
            tem.append(i)
        if Min == Max:
            tem = []

    count = len(tem)

    child1[-1] = child1[-1] - count
    child2[-1] = child2[-1] + count

    cross_point_father = sorted(tem, reverse=True)
    cross_point_mother = sorted(tem, reverse=False)  # 为了不改变原位置
    for i in cross_point_father:
        item = child1.pop(i)
        child1.insert(-1, item)
    for i in cross_point_mother:
        item = child2.pop(i)
        child2.insert(0, item)
    return child1, child2, cross_point


def add_pool(graph, degree_bar):
    node_list = [n for n in range(len(graph.degree())) if graph.degree(n) > degree_bar]
    add_edge = list(itertools.combinations(node_list, 2))
    delete_edge = graph.get_edgelist()
    add = list(set(add_edge).difference(set(delete_edge)))
    return add


def overlap(graph, node1, node2):
    neighbor_1 = graph.neighbors(node1)
    neighbor_2 = graph.neighbors(node2)
    return len(set(neighbor_1).intersection(set(neighbor_2)))


def motif(graph):
    adjacency = graph.get_adjacency()
    motif_mat = np.zeros([graph.vcount(), graph.vcount()])
    motif_mat_norm = np.zeros([graph.vcount(), graph.vcount()])
    nodes = range(graph.vcount())
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if adjacency[i][j] == 1:
                motif_mat[i][j] = overlap(graph, i, j)
                if overlap(graph, i, j) > 0:
                    motif_mat_norm[i][j] = 1
    motif_graph = Graph.Adjacency(motif_mat_norm)
    motif_graph.to_undirected()
    return motif_graph


def degree_distribution(N_degrees):
    # degrees = graph.degree()
    # N_degrees = [x / max(degrees) for x in degrees]
    # Define the bin edges for the intervals
    bin_edges = np.arange(0, 1.1, 0.2)

    # Count the frequency of numbers in each interval
    hist, _ = np.histogram(N_degrees, bins=bin_edges)

    total = sum(hist)
    normalized_hist = [x / total for x in hist]

    # Print the frequency counts
    distribution = normalized_hist
    distribution.reverse()

    epsilon = 1e-3  # Small epsilon value

    # Add epsilon to each probability to avoid zeros
    distribution = np.array(distribution) + epsilon

    # Normalize the distributions to sum up to 1
    distribution = distribution / np.sum(distribution)

    return distribution


def js_divergence(p, q):
    # Ensure that p and q have the same length
    assert len(p) == len(q), "Distributions must have the same length."

    # Check if any probability values are zero in p or q
    if np.any(p == 0) or np.any(q == 0):
        print("Warning: JSD is undefined when a probability is zero.")
        return None

    # Calculate the average distribution
    m = 0.5 * (p + q)

    # Calculate the JSD
    jsd = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

    return jsd


def kl_divergence(p, q):
    # Ensure that p and q have the same length
    assert len(p) == len(q), "Distributions must have the same length."

    # Check if any probability values are zero in p or q
    if np.any(p == 0) or np.any(q == 0):
        print("Warning: KL divergence is undefined when a probability is zero.")
        return None

    # Calculate the KL divergence
    kl_div = np.sum(p * np.log(p / q))

    return kl_div


def randomly_delete_elements(lst, num_elements_to_delete):
    # Make a copy of the list to avoid modifying the original list
    new_list = lst.copy()

    # Ensure the number of elements to delete doesn't exceed the list length
    num_elements_to_delete = min(num_elements_to_delete, len(new_list))

    # Generate a list of random indices to delete
    indices_to_delete = random.sample(range(len(new_list)), num_elements_to_delete)

    # Sort the indices in descending order to avoid index shifting issues during deletion
    indices_to_delete.sort(reverse=True)

    # Initialize a list to store the deleted elements' indices
    deleted_indices = []

    # Delete the elements at the randomly chosen indices and store the deleted indices
    for index in indices_to_delete:
        deleted_indices.append(index)
        del new_list[index]

    return new_list, deleted_indices

def graph_similarity(G1, G2, community1, community2):
    sim = np.zeros((len(community1), len(community2)))
    sim_motif = np.zeros((len(community1), len(community2)))
    G1_motif = motif(G1)
    G2_motif = motif(G2)

    max1_motif = max(G1_motif.degree())
    max2_motif = max(G2_motif.degree())

    max1 = max(G1.degree())
    max2 = max(G2.degree())

    for i, c1 in enumerate(community1):
        distribution1_motif = [x / max1_motif for x in G1_motif.degree(c1)]
        N_distribution1_motif = degree_distribution(distribution1_motif)

        distribution1 = [x / max1 for x in G1.degree(c1)]
        N_distribution1 = degree_distribution(distribution1)

        for j, c2 in enumerate(community2):
            distribution2_motif = [x / max2_motif for x in G2_motif.degree(c2)]
            N_distribution2_motif = degree_distribution(distribution2_motif)

            distribution2 = [x / max2 for x in G2.degree(c2)]
            N_distribution2 = degree_distribution(distribution2)
            sim[i, j] = max(len(c1) / len(c2), len(c2) / len(c1)) * (
                    kl_divergence(N_distribution2, N_distribution1) + kl_divergence(N_distribution1,
                                                                                    N_distribution2)) / 2
            sim_motif[i, j] = max(len(c1) / len(c2), len(c2) / len(c1)) * (
                    kl_divergence(N_distribution2_motif, N_distribution1_motif) + kl_divergence(
                N_distribution1_motif,
                N_distribution2_motif)) / 2
    return sim, sim_motif


def community_alignment(similarity_matrix):
    rows, cols = similarity_matrix.shape
    lower_dimension = min(rows, cols)
    matching = []
    similarity = 0
    for _ in range(lower_dimension):
        max_index = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        similarity = similarity + np.max(similarity_matrix)
        max_row, max_col = max_index
        similarity_matrix[max_row] = -1
        similarity_matrix[:, max_col] = -1
        matching.append(max_index)
    similarity = similarity / lower_dimension

    return similarity, matching

def find_communities(g, method):
    weights = None
    if method == 'fastgreedy':
        return Graph.community_fastgreedy(g.simplify(multiple=True, loops=True, combine_edges=None),
                                          weights).as_clustering()

def calculate_similarity(G1, G2):
    community1 = find_communities(G1, 'fastgreedy')
    community2 = find_communities(G2, 'fastgreedy')
    sim_matrix, sim_motif_matrix = graph_similarity(G1, G2, community1, community2)
    S = np.exp(-sim_matrix)
    S_M = np.exp(-sim_motif_matrix)
    SS = (S + S_M) / 2
    similarity, matching = community_alignment(SS)
    return similarity, matching



def solution_transfer(graph, budget, add_edge, delete_edge, solution, mapping_dic):
    mapped_solutions = []
    edges = graph.get_edgelist()

    for edge_list in solution:
        mapped_edge_list_add = []
        mapped_edge_list_delete = []
        mapped_edge_list = []
        for edge in edge_list:
            if type(edge) == int:
                mapped_edge_list = mapped_edge_list_add + mapped_edge_list_delete
                mapped_edge_list.append(len(mapped_edge_list_add))
                continue
            mapped_node0 = edge[0]
            mapped_node1 = edge[1]
            M_node0 = mapping_dic[mapped_node0]
            M_node1 = mapping_dic[mapped_node1]

            if M_node0 == M_node1:
                if random.random() > 0.5:
                    mapped_edge_list_delete.extend(random.sample(delete_edge, 1))
                else:
                    mapped_edge_list_add.extend(random.sample(add_edge, 1))
            else:
                if (M_node0, M_node1) in edges:
                    mapped_edge_list_delete.append((M_node0, M_node1))
                else:
                    mapped_edge_list_add.append((M_node0, M_node1))

        if len(mapped_edge_list) - 1 < budget:
            quota = budget - len(mapped_edge_list) + 1
            ratio = np.random.randint(0, quota)
            add = random.sample(add_edge, ratio)
            delete = random.sample(delete_edge, quota - ratio)  # 采样p的比例的节点作为增边
            mapped_edge_list = add + mapped_edge_list_add + delete + mapped_edge_list_delete
            mapped_edge_list.append(len(add) + len(mapped_edge_list_add))
        elif len(mapped_edge_list) - 1 > budget:
            quota = len(mapped_edge_list) - 1 - budget
            new_list, deleted_indices = randomly_delete_elements(mapped_edge_list[0:-1], quota)
            count = 0
            for x in deleted_indices:
                if x < len(mapped_edge_list_add):
                    count = count + 1
            mapped_edge_list = new_list
            mapped_edge_list.append(len(mapped_edge_list_add) - count)

        mapped_solutions.append(mapped_edge_list)

    return mapped_solutions


def cal_ARI(labels_true, labels_pred):
    """
    compute ARI
    :param labels_true:
    :param labels_pred:
    :return:
    """
    return metrics.adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)


def evaluate(original_community, pred_community):
    label_original_dict = {}
    label_original = []
    label_pred_dict = {}
    label_pred = []
    for commID, comm in enumerate(pred_community):
        for node in comm:
            label_pred_dict[node] = commID
    for i in range(len(label_pred_dict)):
        label_pred.append(label_pred_dict[i])

    for commID, comm in enumerate(original_community):
        for node in comm:
            label_original_dict[node] = commID
    for i in range(len(label_original_dict)):
        label_original.append(label_original_dict[i])

    ARI = cal_ARI(label_original, label_pred)

    return ARI


def evaluation(graph, result, method):
    graph_copy_1 = graph.copy()
    ratio_1 = result[0][1][-1]
    graph_copy_1.add_edges(result[0][1][:ratio_1])
    graph_copy_1.delete_edges(result[0][1][ratio_1:-1])
    community_1 = find_communities(graph, method)
    c_new_1 = find_communities(graph_copy_1, method)
    nmi = compare_communities(community_1, c_new_1, method='nmi', remove_none=False)
    vi = compare_communities(community_1, c_new_1, method='vi', remove_none=False)
    split = compare_communities(community_1, c_new_1, method='split-join', remove_none=False)
    ARI = evaluate(community_1, c_new_1)

    return [nmi, vi, split, ARI]


def filter_matrix(similarity_matrix, index):
    # Define the indices of rows and columns you want to keep
    size = len(similarity_matrix)
    for i in range(size):
        for j in range(size):
            if j not in index[i]:
                similarity_matrix[i][j] = 0

    return similarity_matrix


def entropy(c):
    result = -1
    if len(c) > 0:
        result = 0
    for x in c:
        if x == 0:
            continue
        result += (-x) * math.log(x, 2)
    return result


def confusion_value(c, c1):
    node_number = 0
    for i in range(len(c)):
        node_number = node_number + len(c[i])

    M = np.ones((len(c), len(c1)))
    M1 = np.ones((len(c), len(c1)))
    M2 = np.ones((len(c), len(c1)))
    for i in range(len(c)):
        for j in range(len(c1)):
            M[i][j] = len(list(set(c[i]).intersection(set(c1[j]))))

    for i in range(len(c)):
        M1[i] = M[i] / len(c[i])

    for i in range(len(c1)):
        M2[:, i] = (M[:, i] / len(c1[i])) * math.log(len(c1[i]))
    c_entropy = 0

    for i in range(len(c)):
        c_entropy = (len(c[i]) / node_number) * entropy(M1[i]) + c_entropy

    objecitve2 = math.exp(-np.max(M2))

    return c_entropy * objecitve2


def improved_rate(edges_old, edge_new, edges_transfer):
    edge_old = [element for sublist in edges_old for element in sublist[0:-1]]
    edge_new = [element for sublist in edge_new for element in sublist[0:-1]]
    edges_transfer = [element for sublist in edges_transfer for element in sublist[0:-1]]

    improved_edge = set(edge_new) - set(edge_old)

    intersection = improved_edge.intersection(set(edges_transfer))

    count = 0
    rate = 0
    if len(intersection) > 0:
        for transfer_edges in intersection:
            for edge in edge_new:
                if edge == transfer_edges:
                    count = count + 1
        rate = count / len(edge_new)

    return rate


def normalize_matrix(cofficient_matrix):
    cofficient_matrix = list(cofficient_matrix)
    for i in range(len(cofficient_matrix)):
        cofficient_matrix[i][i] = 0
    normalized_list = []
    for row in cofficient_matrix:
        row_sum = sum(row)
        normalized_row = [value / row_sum for value in row]
        normalized_list.append(normalized_row)
    return normalized_list


def turn_matrix(cofficient_matrix):
    normalized_list = []
    for row in cofficient_matrix:
        normalized_row = [1 - value for value in row]
        for i, value in enumerate(normalized_row):
            if value == 1:
                normalized_row[i] = 0
        normalized_list.append(normalized_row)
    return normalized_list

def graph_read(GrapH):
    if GrapH == 'dolphins':
        graph = Graph.Read_GML("CSD dataset/dolphins/dolphins.gml")
    elif GrapH == 'lesmis':
        graph = Graph.Read_GML("CSD dataset/lesmis/lesmis.gml")
    elif GrapH == 'netscience':
        graph = Graph.Read_GML("CSD dataset/netscience/netscience.gml")
        cl = graph.clusters()
        graph = cl.giant()
    elif GrapH == 'polbooks':
        graph = Graph.Read_GML("CSD dataset/polbooks/polbooks.gml")
    elif GrapH == 'adjnoun':
        graph = Graph.Read_GML("CSD dataset/adjnoun/adjnoun.gml")
    elif GrapH == 'Erods':
        graph = Graph.Read_Ncol("CSD dataset/erdos.txt").as_undirected()
    elif GrapH == 'USAir':
        graph = Graph.Read_GML("CSD dataset/USAir.gml")
    elif GrapH == 'bio_celegans':
        graph = Graph.Read_Ncol("CSD dataset/bio-celegans.txt").as_undirected()
    return graph