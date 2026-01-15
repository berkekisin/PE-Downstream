import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
import os 
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset

from GNNPlus.transform.add_dinov_pe import AddDinovPE

def save_embeddings(dataset, emb_name, emb_folder):
    os.makedirs(os.path.normpath(emb_folder), exist_ok=True)
    emb = getattr(dataset.data, emb_name)
    slices = dataset.slices[emb_name]
    torch.save(emb, os.path.join(emb_folder, 'data.pt'))
    torch.save(slices, os.path.join(emb_folder, 'slices.pt'))
    
def precompute_dino(cfg, dataset, attr_name = 'pestat_Dino'):
    dataset_folder = cfg.dataset.dir
    dataset_name = cfg.dataset.name
    emb_save_folder = os.path.join(dataset_folder, dataset_name, cfg.posenc_Dino.save_folder )
    
    # load precomputed emb
    if os.path.isdir(emb_save_folder):
        print('Found precomputed dino embedding!')
        emb = torch.load(os.path.join(os.path.normpath(emb_save_folder), 'data.pt'), map_location="cpu")
        slices = torch.load(os.path.join(os.path.normpath(emb_save_folder), 'slices.pt'), map_location="cpu")
        #setattr(dataset, attr_name, emb)
        dataset.data[attr_name] = emb
        dataset.slices[attr_name] = slices
        data_list = []
        for i in tqdm(range(len(dataset)), desc='Embeddings not found precomputing!'):
            data = dataset.get(i) 
            data_list.append(data)
            
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)    
    else:
        transform_func = AddDinovPE(model_name=cfg.posenc_Dino.model_name,
                                attr_name=attr_name,
                                aggr=cfg.posenc_Dino.aggr)
        data_list = []
        for i in tqdm(range(len(dataset)), desc='Embeddings not found precomputing!'):
            data = dataset.get(i) 
            data = transform_func(data)
            data_list.append(data)
            
            
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        
        save_embeddings(dataset, emb_name=attr_name, emb_folder=emb_save_folder)
        
    return dataset

def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset
     
    if transform_func.func.__name__ == 'compute_posenc_stats':
        if 'Dino' in transform_func.keywords['pe_types']:
            cfg = transform_func.keywords['cfg']
            precompute_dino(cfg, dataset)
    else:
        data_list = [transform_func(dataset.get(i))
                        for i in tqdm(range(len(dataset)),
                                    disable=not show_progress,
                                    mininterval=10,
                                    miniters=len(dataset)//20)]
        data_list = list(filter(None, data_list))

        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        
    return


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        # logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data
