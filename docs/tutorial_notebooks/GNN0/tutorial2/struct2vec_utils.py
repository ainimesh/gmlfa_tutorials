import numpy as np
import networkx as nx
from random import random
from fastdtw import fastdtw
from gensim.models.word2vec import Word2Vec

def custom_distance(x, y):
    # Compute the ratio (max / min) - 1 for distance
    if min(x, y) != 0:
        return (max(x, y) / min(x, y)) - 1
    else:
        return float('inf')

def DTW_Distance(sequence_1, sequence_2):
    if sequence_1 == [] or sequence_2 == []:
        return np.inf
        
    seq1_tuples = [(x,) for x in sequence_1]
    seq2_tuples = [(x,) for x in sequence_2]

    distance, path = fastdtw(seq1_tuples, seq2_tuples, dist=custom_distance)
    return distance[0]

def get_k_hop_degree_sequence(G, node, k):
    if k == 0:
        return [G.degree(node)]  # Return the degree of the node itself
    
    current_level_nodes = {node}

    for i in range(k):
        next_level_nodes = set()
        for n in current_level_nodes:
            next_level_nodes.update(G.neighbors(n))

        current_level_nodes = next_level_nodes
    
    degree_seq = sorted([G.degree(neighbor) for neighbor in current_level_nodes])
    return degree_seq

def initialize_probs(G, num_levels):
    """
    Initialize the probs dictionary for transition probabilities and edge weights.
    
    """
    
    probs = {}
    
    for node in G.nodes():
        probs[node] = {}
        for level in range(num_levels):
            # Initialize transition probabilities for each level
            probs[node][level] = {
                'transition': np.zeros(2),  # Transition probabilities: [up, down]
                'same': np.zeros(len(G.nodes()))  # Transition probabilities to the same level
            }
    
    return probs

def compute_probabilities(G, num_levels):
    probs = initialize_probs(G, num_levels = num_levels)
    f = {node: {other: [0] * num_levels for other in G.nodes()} for node in G.nodes()}
    
    for level in range(num_levels):
        for node in G.nodes():
            # Calculate degree sequences and DTW distances
            degree_seq = get_k_hop_degree_sequence(G,node,level)
            
            for other in G.nodes():
                if node == other:
                    continue
                
                other_degree_seq = get_k_hop_degree_sequence(G,other,level)
                # Compute DTW distance
                dtw_distance = DTW_Distance(degree_seq, other_degree_seq)
                if level == 0:
                    f[node][other][level] = dtw_distance
                else:
                    f[node][other][level] = f[node][other][level-1] + dtw_distance

    # Compute edge weights and transition probabilities
    edge_weights = {}
    for level in range(num_levels):
        edge_weights[level] = {}
        for node in G.nodes():
            for other in G.nodes():
                if node != other:
                    edge_weights[level][(node, other)] = np.exp(-f[node][other][level])

    for level in range(num_levels):
        for node in G.nodes():
            # Compute weights to the same node at level k+1 and k-1
            count = np.sum([1 for other in G.nodes() if edge_weights[level].get((node, other), 0) > np.mean(list(edge_weights[level].values()))])
            weight_same_node_up = np.log(count + np.e)
            weight_same_node_down = 1
            
            move_up_probability = weight_same_node_up / (weight_same_node_up + weight_same_node_down)
        
            if level == 0:
                probs[node][level]["transition"][0] = 1
                probs[node][level]["transition"][1] = 0
            elif level == num_levels-1:
                probs[node][level]["transition"][0] = 0
                probs[node][level]["transition"][1] = 1
            else:
                probs[node][level]["transition"][0] = move_up_probability
                probs[node][level]["transition"][1] = 1 - move_up_probability
        
            
            # Compute transition probabilities to other nodes in the same level
            total_weight = np.sum([edge_weights[level].get((node, other), 0) for other in G.nodes() if node != other])
            for other in G.nodes():
                if node != other:
                    weight = edge_weights[level].get((node,other),0)
                    if total_weight > 0:
                        probs[node][level]['same'][other] = weight/total_weight
                    else:
                        probs[node][level]['same'][other] = weight

    return probs

def generate_random_walks(G, probs, max_walks, walk_len, jump_probability):
    walks = list()
    for start_node in G.nodes():
        for i in range(max_walks):
            level = 0
            walk = [start_node]
            for k in range(walk_len-1):
                current_node = walk[-1] # current node
                if random()<jump_probability: # then we transition to another level
                    options = [level+1, level-1]
                    probabilities = probs[current_node][level]["transition"]
                    level = np.random.choice(options, p=probabilities)
                    next_node = current_node # in the new level
                    walk.append(next_node)
                else: # then we stay in the current level
                    options = list(G.nodes)
                    probabilities = probs[current_node][level]["same"]
                    next_node = np.random.choice(options, p=probabilities)
                    walk.append(next_node)
                
                current_node = next_node
                
            walks.append(walk)

    np.random.shuffle(walks)
    walks = [list(map(str,walk)) for walk in walks]
    
    return walks


def Struct2Vec(graph, dimensions = 32, walk_length = 40, num_walks = 20, window = 10, Q = 0.1, max_levels = None):
    """
    Custom struct2vec implementation
    Inputs:
    graph : a networkx graph
    dimensions: node embedding dimension
    walk_length : length of generated biased random walks
    num_walks: walks generated per node
    window: skipgram window size
    Q: probability of jumping to another level
    max_levels: optional

    Outputs:
    Node embedding dictionary
        
    """
    if max_levels is None:
        max_levels = int(nx.diameter(graph)/2)
    max_levels = max(max_levels,2)
    probabilities = compute_probabilities(graph, num_levels=max_levels)
    walks = generate_random_walks(graph, probs=probabilities, max_walks=num_walks, walk_len=walk_length, jump_probability=Q)
    model = Word2Vec(sentences=walks, window=window, vector_size=dimensions)
    return model.wv
    
