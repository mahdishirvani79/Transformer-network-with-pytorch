import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct, self).__init__()
        self.activation = nn.Softmax(dim = -1)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        x = torch.bmm(Q, K.transpose(-1, -2))
        dk = torch.tensor(K.size(-1))
        x = x.div(torch.sqrt(dk))
        x = self.activation(x)
        x = torch.bmm(x, V)
        return x



def test_scaled_dot_product():
    batch_size = 2
    sequence_number = 5
    d_k = 4
    d_v = 6
    Q = torch.full((batch_size, sequence_number, d_k), 1, dtype= torch.float)
    K = torch.full((batch_size, sequence_number, d_k), 2, dtype= torch.float)
    V = torch.full((batch_size, sequence_number, d_v), 3, dtype= torch.float)
    scaled_dot_product = ScaledDotProduct()
    product = scaled_dot_product(Q, K, V)
#     torch_versio = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    print(product)
#     print(torch_versio)



class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model):
        super(MultiHeadAttention, self).__init__()
        self.dk = 10
        self.dv = 12
        self.num_head = num_head
        self.WQ = nn.Parameter(torch.randn(self.num_head, d_model, self.dk))
        self.WK = nn.Parameter(torch.randn(self.num_head, d_model, self.dk))
        self.WV = nn.Parameter(torch.randn(self.num_head, d_model, self.dv))
        self.WO = nn.Parameter(torch.randn(self.num_head * self.dv, d_model))
        self.reset_parameters()
        self.scaled_dot_product = ScaledDotProduct()
        
    
    def reset_parameters(self):
        nn.init.xavier_uniform(self.WQ)
        nn.init.xavier_uniform(self.WK)
        nn.init.xavier_uniform(self.WV)
        nn.init.xavier_uniform(self.WO)
        
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        heads = list()
        for i in range(self.num_head):
            WQi, WKi, WVi = self.WQ[i, :, :], self.WK[i, :, :], self.WV[i, :, :]
            q = torch.bmm(Q, WQi.unsqueeze(0).repeat(Q.size(0), 1, 1))
            k = torch.bmm(K, WKi.unsqueeze(0).repeat(Q.size(0), 1, 1))
            v = torch.bmm(V, WVi.unsqueeze(0).repeat(Q.size(0), 1, 1))
            heads.append(self.scaled_dot_product(q,k,v))
        out = torch.cat(heads, dim=-1)
        out = torch.bmm(out, self.WO.unsqueeze(0).repeat(Q.size(0), 1, 1))
        return out



def test_multi_head_attention():
    num_head = 8
    d_model = 128
    batch_size = 2
    sequence_number = 5
    multi_head_attention = MultiHeadAttention(num_head, d_model)
    Q = torch.full((batch_size, sequence_number, d_model), 1, dtype= torch.float)
    K = torch.full((batch_size, sequence_number, d_model), 2, dtype= torch.float)
    V = torch.full((batch_size, sequence_number, d_model), 3, dtype= torch.float)
    out = multi_head_attention(Q, K, V)
    print(out)
    # use torch version to see if we were correct
    torch_multi_head_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_head, batch_first=True)
    torch_out = torch_multi_head_attention(Q, K, V, need_weights=False)
    print(torch_out[0])
    print(torch.eq(out, torch_out[0]).all())