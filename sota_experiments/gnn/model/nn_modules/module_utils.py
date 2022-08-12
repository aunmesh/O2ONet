from torch_geometric.nn import TransformerConv, GCN, GATConv

def get_gnn(config, in_dim, out_dim):
    
    if config['GNN'] == 'TransformerConv':
        return TransformerConv(in_dim, out_dim)
    
    if config['GNN'] == 'GCN':
        return GCN(in_dim, out_dim)
    
    if config['GNN'] == 'GATConv':
        return GATConv(in_dim, out_dim)