import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder

@register_node_encoder('Dinov')
class DinoEncoder(torch.nn.Module):

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        self.pestat_var =  'pestat_Dino'
        pecfg = getattr(cfg, f"posenc_Dino")
        dim_pe_in = pecfg.dim_pe_in
        dim_pe_out = pecfg.dim_pe
        
        
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe_out < 0: 
            raise ValueError(f"PE dim size {dim_pe_out} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe_out > 0:
            self.linear_x = nn.Linear(dim_pe_in, dim_emb - dim_pe_out)
        self.expand_x = expand_x and dim_emb - dim_pe_out > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(dim_pe_in)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(dim_pe_in, dim_pe_out))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe_in, 2 * dim_pe_out))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe_out, 2 * dim_pe_out))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe_out, dim_pe_out))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(dim_pe_in, dim_pe_out)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):

        if not hasattr(batch, self.pestat_var):
            raise ValueError(f"Precomputed '{self.pestat_var}' variable is "
                             f"required for {self.__class__.__name__}")

        pos_enc = getattr(batch, self.pestat_var)  
    
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch
