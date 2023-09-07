import torch


# datastructure needed in data for NENN

# edge index vector(filtered) - to look up node based nighbors of an edge
# line adj mat - to look up edge based neighbors of edge e

# adj mat - to look up node based neighbors of a node
#  edge_index_vector  - to look up edge based neighbors of a node - this look up can be vectorise


# what do we have currently - edge_index_vector, edge_feature_matrix, node_feature_matrix, 
# num of nodes, num_of edges

# what do I need to do - I need to make the line adj mat, filtered edge 


def filter_edge_index(edge_index, device):
    # don't remove self loops
    # remove repeating edges - but why - so that same 2 edges are not repeated again and again
    
    all_edges = []
    num_edges = edge_index.shape[0]
    
    for e in range(num_edges):

        node0, node1 = edge_index[:,e]

        if node0 != node1:
            if node0 < node1:
                if [node0, node1] not in all_edges:
                    all_edges.append([node0, node1])
            else:
                if [node1, node0] not in all_edges:
                    all_edges.append([node1, node0])
    
        if node0 == node1:
            if node0 >= 0:
                if [node0, node1] not in all_edges:
                    all_edges.append([node0, node1])
    
    all_edges_tensor = torch.tensor(all_edges, device=device)
    return all_edges_tensor



def make_adjacency_matrix(edge_index, num_nodes, device):
    
    adj_mat = torch.zeros(num_nodes, num_nodes, dtype=bool ,device=device)
    num_edges = edge_index.shape[0]
    
    for n in range(num_edges):
        node1, node2 = edge_index[:,n]
        adj_mat[node1, node2] = True
        adj_mat[node2, node1] = True
    
    return adj_mat


def make_line_graph_adj_mat(edge_index, adj_mat):
    # edge index is the filtered edge index
    # make adjacency matrix according to the number of edges
        
    num_edges = edge_index.shape[1]
    
    line_adj_mat = torch.zeros(num_edges, num_edges, dtype=bool ,device=device)
    for e in range(num_edges):
        node0, node1 = edge_index[:, e]
        node0_neighbors = find_node_based_nbor_of_node(adj_mat, node0)
        node1_neighbors = find_node_based_nbor_of_node(adj_mat, node1)
        
        temp_edge_neighbors = find_edge_based_nbor_of_edge( [node0, node1], 
                                                           [node0_neighbors, node1_neighbors], edge_index)
        for t in temp_edge_neighbors:
            line_adj_mat[e, t] = True
            line_adj_mat[t, e] = True
    return line_adj_mat

            
def find_node_based_nbor_of_node(adj, i):
    # req_row = adj[i]
    req_neighbors = torch.where(adj[i] > 0)[0].tolist()
    return req_neighbors


def find_edge_based_nbor_of_edge( node_list, node_nbor_list, edge_index):

    edges_to_look_for = []
    
    for i in range(2):
        curr_node = node_list[i]  # one of the 2 nodes making the edge
        curr_nbor_list = node_nbor_list[i]  # the neighbors of the corresponding node
            
        for i , n in enumerate(curr_nbor_list):
            if(n<= curr_node):
                edges_to_look_for.append([n, curr_node])
            else:
                edges_to_look_for.append([curr_node, n])
    
    edge_based_neighbors = []
    
    num_edges = edge_index.shape[1]

    for i, temp_e in enumerate(edges_to_look_for):
        temp_n0, temp_n1 = temp_e

        for n in range(num_edges):
            if temp_n0 == edge_index[0, n] and temp_n1 == edge_index[1, n]:
                edge_based_neighbors.append(n)

    return edge_based_neighbors