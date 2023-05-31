import os
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull
from ogb.linkproppred import PygLinkPropPredDataset


data_dir = './data'
datasets = ['Cora', 'PubMed', 'DBLP']

def get_stat(d):
    dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    data = dataset[0]
    print(d)
    print('Number of nodes:', data.num_nodes)
    print('Number of edges:', data.num_edges)
    print('Number of max deleted edges:', int(0.05 * data.num_edges))
    if hasattr(data, 'edge_type'):
        print('Number of nodes:', data.edge_type.unique().shape)

def main():
    for d in datasets:
        get_stat(d)   

if __name__ == "__main__":
    main()
