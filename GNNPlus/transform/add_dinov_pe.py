from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms.add_positional_encoding import add_node_attr
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected, get_laplacian
from torch_scatter import scatter_sum, scatter_mean
from src.pretrained.dinov import load_llama_model, load_gpt2_model, load_qwen2_model, load_mixtral_moe_model, DINOModel
from transformers import PretrainedConfig

@functional_transform('add_dinov_pe')
class AddDinovPE(BaseTransform):
    def __init__( self, model_name, attr_name: Optional[str] = 'pestat_Dino', aggr='sum'):
        context_size = 1024
        vocab_size = 20000 + 3
        config = load_llama_model(vocab_size, context_size)
        model = DINOModel.from_pretrained(model_name, config=config, token=None)
        self.embedding_model = model.eval()
        self.attr_name = attr_name
        self.aggr = aggr
    
    def compute_embedding(self, edge_index):
        bos_token = self.embedding_model.config.bos_token_id
        eos_token = self.embedding_model.config.eos_token_id
        edge_index_flatt =  edge_index.T.flatten() 
        input = torch.tensor([bos_token] + edge_index_flatt.tolist() + [eos_token]).view(1,-1)
    
        hidden_states = self.embedding_model(input)
        #print(hidden_states.shape)

        hidden_states = [out.clone().detach() for out in hidden_states.hidden_states]
        embedding_tensors = [hidden_states[i][0][1:-1,:] for i in range(len(hidden_states))]
        final_tensor = torch.zeros_like(embedding_tensors[-1])
        for i in range(len(hidden_states)):
            final_tensor += embedding_tensors[i]
        
        if self.aggr == 'mean':
            return scatter_mean(final_tensor, edge_index_flatt, dim=0)
        elif self.aggr == 'sum':
            return scatter_sum(final_tensor, edge_index_flatt, dim=0)


        #embedding_tensors = [hidden_states[i][1:-1].clone().detach() for i in range(len(hidden_states))]
        #embedding_tensors = [hidden_states[i][0][1:-1,:] for i in range(len(hidden_states))]
        #final_tensor = torch.zeros_like(embedding_tensors[-1])
        #for i in range(len(hidden_states)):
        #    final_tensor += embedding_tensors[i]
        
        #if self.aggr == 'mean':
        #    return scatter_mean(final_tensor, edge_index_flatt, dim=0)
        #elif self.aggr == 'sum':
        #    return scatter_sum(final_tensor, edge_index_flatt, dim=0)
            
       
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        N = data.num_nodes
        assert N is not None

        if data.is_undirected:
            undir_edge_index = data.edge_index
        else:
            undir_edge_index = to_undirected(data.edge_index)
            
        data.edge_index = undir_edge_index
        embedding = self.compute_embedding(undir_edge_index)
        data = add_node_attr(data, embedding, attr_name=self.attr_name)

        return data