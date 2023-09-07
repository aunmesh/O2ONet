import torch


def concatenate(t1, t2):
    return torch.cat((t1, t2))


def mean(t1, t2):
    return t1.add(t2)/2.0


def aggregate(t1, t2, name):
    
    assert name == "mean" or name == "concat", "aggregator name can only be mean or concat"

    if name=='mean':
        return mean(t1, t2)
    
    if name=='concat':
        return concatenate(t1, t2)


def make_classifier_inputs(collated_node_embeddings, pairs, classifier_input_dimension, device, aggregation='mean'):
    '''
    makes the classifier input from the node embeddings and pairs
    args:
        node_embeddings: Embeddings of the various nodes

        pairs: A tensor of shape [b_size, MAX_PAIRS, 2]
            b_size is batch size, MAX_PAIRS is the maximum no. of pairs
    
    returns:
        node
    '''

    num_batches = collated_node_embeddings.shape[0]
    num_pairs = pairs.shape[1]   # Always equal to max pairs

    # classifier_input is the tensor which will be passed to the fully connected classifier
    # for feature classification
    classifier_input = torch.empty(
        num_batches, num_pairs, classifier_input_dimension, device=device)

    for b in range(num_batches):

        for i in range(num_pairs):

            ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]

            emb0 = collated_node_embeddings[b, ind0]
            emb1 = collated_node_embeddings[b, ind1]
            classifier_input[b, i] = aggregate(emb0, emb1, aggregation)

    return classifier_input



def collate_node_features(node_features, num_nodes, device):
    
    # dimension_dict signifies which dimensions correspond to batch, feature or node
    # so for example if data is of shape [batch, num_nodes, node_feature] then 
    # dimension dict will have the default value above.

    total_nodes = int(torch.sum(num_nodes))
    b_size, _, dim_features = node_features.shape

    res = torch.zeros((total_nodes, dim_features), dtype = node_features.dtype, device=device)
    
    curr_index = 0
    node_slices = torch.zeros((total_nodes), dtype=torch.long, device=device)

    for b in range(b_size):
        
        curr_num_nodes = int(num_nodes[b])
        
        temp = node_features[b,:curr_num_nodes,:]

        res[curr_index: curr_index + curr_num_nodes,:] = temp
        node_slices[curr_index:curr_index + curr_num_nodes] = b

        curr_index+=curr_num_nodes
    
    return res, node_slices


def collate_edge_indices(edge_index, num_edges, num_objects, device):

    total_edges = int(torch.sum(num_edges))
    res = torch.zeros((2, total_edges), dtype = edge_index.dtype, device=device)

    b_size = int(edge_index.shape[0])

    curr_index = 0
    edge_slices = torch.zeros((total_edges), dtype=torch.long, device=device)

    lower = 0
    upper = 0

    for b in range(b_size):

        curr_num_edges = int(num_edges[b])
        temp_vec = edge_index[b, :,:curr_num_edges]

        temp_node_offset = torch.sum(num_objects[lower:upper])

        res[ : , curr_index: curr_index+curr_num_edges] = temp_node_offset
        res[ : , curr_index: curr_index+curr_num_edges] += temp_vec

        edge_slices[curr_index:curr_index+curr_num_edges] = b

        upper+=1
        curr_index+=curr_num_edges

    return res, edge_slices


def decollate_node_embeddings(all_node_embeddings, node_slicing, device, pad_len=15):

    dim_embedding = all_node_embeddings.shape[-1]
    b_size = int(node_slicing[-1]+1)
    
    result = torch.zeros((b_size, pad_len, dim_embedding), dtype=all_node_embeddings.dtype, device=device)
    
    for i in range(b_size):
        curr_obj = i
        
        indices = torch.where(node_slicing == curr_obj)
        num_obj = indices[0].shape[0]

        temp_embeddings = all_node_embeddings[indices]
        result[i][:num_obj] = temp_embeddings

    return result