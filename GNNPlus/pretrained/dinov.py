import torch
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch.distributed as dist
import math 
import numpy as np

import os
import numpy as np
from transformers import (
    GPT2Config,
    LlamaConfig,
    Qwen2Config,
    MixtralConfig,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import tqdm
import torch.nn as nn
import torch.nn.functional as F


import torch
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch.distributed as dist
import math 
import numpy as np

import os
import numpy as np
from transformers import (
    GPT2Config,
    LlamaConfig,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

import tqdm
import torch.nn as nn
import torch.nn.functional as F


def shuffle_edge_index(edge_index, bos_token, eos_token):

    #create dict of nodes and their new ones
    max = torch.topk(edge_index, k=3)[3].item() #since bos and eos are the highest values
    min = torch.min(edge_index).item()

    dict_a = np.arange(min, max+1)
    dict_b = np.arange(min, max+1).shuffle()
    #allocate the new index

    index_dict = dict(zip(dict_a, dict_b))
    index_dict[bos_token] = bos_token
    index_dict[eos_token] = eos_token

    

    #return index 
    return 



def cosine_scheduler(iteration, base_value, final_value, total_iters, warmup_iters=0, start_warmup_values=0, freeze_iters=0):

    if iteration >= total_iters:
        return final_value

    if iteration < freeze_iters:
        return 0.0

    if iteration < warmup_iters + freeze_iters:
        warmup_progress = (iteration - freeze_iters) / max(1, warmup_iters)
        return start_warmup_values + (base_value - start_warmup_values) * warmup_progress

    cosine_iteration = iteration - warmup_iters
    cosine_total = total_iters - freeze_iters - warmup_iters
    cosine_progress = cosine_iteration / max(1, cosine_total)
    return final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * cosine_progress))


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, bottleneck_dim):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            #nn.BatchNorm1d(hidden_dim//2),
            #nn.Identity(), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias = True),
            # nn.BatchNorm1d(hidden_dim//2),
            #nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim, bias=True),
        )

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        

    def forward(self, x):
        x = self.mlp1(x)   
        #print(x.isnan().any())
        eps =1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, eps=eps, p=2)
        #print(x.isnan().any())
        x = self.last_layer(x)
        return x


""" class DINOLoss(nn.Module):

    def __init__(self, student_temp, center_momentum, teacher_temp, out_dim, warmup_temp, warmup_epochs, total_epochs):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim).to(device='cuda'))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_temp, teacher_temp, warmup_epochs),
            np.ones(total_epochs - warmup_epochs) * teacher_temp
        ))

    @torch.no_grad()
    def teacher_center_softmax(self, t):
        return F.softmax(10* torch.tanh((t - self.center) / self.teacher_temp), dim=-1)

    def forward(self, s1, t1,s2,t2, epoch):
        s1_out = s1 / self.student_temp
        s2_out = s2 / self.student_temp
        #print(s1_out)
        #print(s2_out)
        #print(t1.shape)
        self.teacher_temp = self.teacher_temp_schedule[math.floor(epoch)]
        #print(((t1 - self.center) / teacher_temp).shape)
        #t1_shift = ((t1 - self.center) / teacher_temp) #- torch.max(t1,dim=-1, keepdim=True)[0]
        #t2_shift = ((t2 - self.center) / teacher_temp) #- torch.max(t2,dim=-1, keepdim=True)[0]
        #t1_out = nn.functional.softmax(10*torch.tanh((t1 - self.center) / teacher_temp), dim=-1)
        #t1_out = F.log_softmax(t1_shift, dim=-1)
        #t1_out = t1_out.detach()
        #t2_out = nn.functional.softmax(10*torch.tanh((t2 - self.center) / teacher_temp), dim=-1)
        #t2_out = F.log_softmax(t2_shift, dim=-1)
        #t2_out = t2_out.detach()
        t1_out = self.teacher_center_softmax(t1).detach()
        t2_out = self.teacher_center_softmax(t2).detach()
        #loss = torch.sum(((t1 - self.center) / teacher_temp).detach() * s2_out, dim=-1)/2 + torch.sum(((t2 - self.center) / teacher_temp).detach() * s1_out, dim=-1)/2
        loss = -torch.sum(t1_out * F.log_softmax(s2_out, dim=-1),dim=-1)/2 + -torch.sum(t2_out * F.log_softmax(s1_out, dim=-1),dim=-1)/2
        #print(loss)
        self.update_center(torch.sum(torch.cat((t1, t2), dim=0), dim=0, keepdim=True))
        return loss.mean()
    
    @torch.no_grad()
    def update_center(self, teacher_output):

        Update center used for teacher output.

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output)) #* dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class CustomLossTrainerDINO(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)

    
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        s1,t1,s2,t2 = model(**inputs)
        #logits = outputs.get("logits")
        
        # Compute the custom loss using your loss function.
        criterion = DINOLoss(0.1, out_dim=128, teacher_temp=0.07, warmup_temp=0.04, warmup_epochs=int(0.1*self.args.num_train_epochs), total_epochs=self.args.num_train_epochs, center_momentum=0.9)
        loss = criterion(s1,t1,s2,t2, self.state.epoch)
        
        return (loss, (s1,t1,s2,t2)) if return_outputs else loss

    def update_teacher(self, momentum):
        with torch.no_grad():
            for param_q, param_k in zip(self.model.student.parameters(), self.model.teacher.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1 - momentum)

    def training_step(self, model, inputs, num_items_in_batch=None):
        momentum = cosine_scheduler(
            iteration=self.state.global_step,
            base_value=self.model.config.momentum_teacher,
            final_value=self.model.config.final_momentum_teacher,
            total_iters=self.state.max_steps,
        )

        ret = super(CustomLossTrainerDINO, self).training_step(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch,
        )
        self.update_teacher(momentum)
        return ret """


def load_gpt2_model(vocab_size):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_size,
        n_ctx=context_size,
        n_embd=640,
        n_layer=12,
        n_head=10,
        eos_token_id=vocab_size-1,
        pad_token_id=vocab_size-1,
    )
    model = GPT2LMHeadModel(config)
    return model


def load_llama_model(vocab_size, context_size):
    config = LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=context_size,
        hidden_size=384,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        # hidden_size=1024,
        # intermediate_size=2730,
        # num_hidden_layers=12,
        # num_attention_heads=16,
        bos_token_id=vocab_size-3,
        eos_token_id=vocab_size-2,
        pad_token_id=vocab_size-1,
        use_cache=False,
        momentum_teacher=0.996,
        final_momentum_teacher=1.0,
        hidden_size_mlp=512,
        bottleneck_dim=64,
        output_dim=128,
        student_temp=0.1

    )
    #model = LlamaForCausalLM(config)
    return config#model

def load_qwen2_model(vocab_size, context_size):
    config = Qwen2Config(
        vocab_size=vocab_size,
        max_position_embeddings=context_size,
        hidden_size=384,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        # hidden_size=1024,
        # intermediate_size=2730,
        # num_hidden_layers=12,
        # num_attention_heads=16,
        use_sliding_window_attention=True,
        num_key_value_heads=2,
        sliding_window_size=512,
        max_window_layers=2,
        bos_token_id=vocab_size-3,
        eos_token_id=vocab_size-2,
        pad_token_id=vocab_size-1,
        use_cache=False,
        momentum_teacher=0.996,
        final_momentum_teacher=1.0,
    )
    return config

def load_mixtral_moe_model(vocab_size, context_size):
    config = MixtralConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=context_size,
        bos_token_id=vocab_size-3,
        eos_token_id=vocab_size-2,
        pad_token_id=vocab_size-1,
        sliding_window=512,
        num_experts_per_tok=2,
        num_local_experts=4,
        momentum_teacher=0.996,
        final_momentum_teacher=1.0,
    )

    return config

class DINOModel(PreTrainedModel):
    """
    SimSiam model built on top of a Hugging Face transformer backbone.
    """
    def __init__(self, config):
        super().__init__(config)
        # Load the transformer backbone
        self.teacher = AutoModel.from_config(config)
        #self.student = AutoModel.from_config(config)

        #self.student_head = DINOHead(in_dim=config.hidden_size, hidden_dim=config.hidden_size_mlp, bottleneck_dim=config.bottleneck_dim, out_dim=config.output_dim)
        self.teacher_head = DINOHead(in_dim=config.hidden_size, hidden_dim=config.hidden_size_mlp, bottleneck_dim=config.bottleneck_dim, out_dim=config.output_dim)

        #for param_src, param_dst in zip(self.student.parameters(),
        ##                                self.teacher.parameters()):
        #    param_dst.data.copy_(param_src.data)
        #    param_dst.requires_grad = False

        self.student_temp = config.student_temp
        self.output_dim = config.output_dim
        self.config = config
        #self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Pass input through the transformer backbone
        #inputs_2 = self.random_edge_removal(input_ids)
        #print(torch.isnan(inputs_2).any())
        #print(torch.isnan(input_ids).any())
        output_teacher_1= self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        #hidden_states = [out.clone().detach() for out in output_teacher_1.hidden_states]
        #return_states = [self.teacher_head(out) for out in hidden_states]
        
        '''
        output_teacher_2 = self.teacher(
            input_ids=inputs_2,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output_students_1 = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output_student_2 = self.student(
            input_ids=inputs_2, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
         '''
    
        return output_teacher_1 #self.teacher_head(output_teacher_1.last_hidden_state)
       
       
    def random_edge_removal(self, input_ids):
        """
        Randomly remove edges from the input_ids.
        """
        # Flatten the input tensor and randomly select an index to remove
        flattened = input_ids
        
        if flattened.size(0) > 0:
            random_index = np.random.randint(0, (flattened.size(0)//2)-2)#torch.randint(0, (flattened.size(0)//2)-2, (1,)).item()
            flattened[:, random_index:random_index+2 ] = self.config.pad_token_id
            flattened[:, random_index*2:random_index*2+2 ] = self.config.pad_token_id
            #flattened = torch.cat((flattened[:,:random_index], flattened[:,random_index + 2:]),dim=1)
        return flattened

    
# Example configuration and model initialization
if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    torch._dynamo.config.capture_scalar_outputs = True


    context_size = 1024
    vocab_size = 20000 + 3 #including padding token
    # Load a configuration for the transformer backbone
    #config = load_llama_model(vocab_size, context_size)
    
    # Initialize the SimSiam model