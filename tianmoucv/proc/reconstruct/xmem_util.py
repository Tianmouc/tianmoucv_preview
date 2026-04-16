#copy from xmem project
import math
import numpy as np
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F


def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys(query,mem)
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys(key,feature)
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None
    
    mk = mk.permute(0,2,1)
    qk = qk.permute(0,2,1)
    #permute it

    # similar to STCN if we don't have the selection term
    a_sq = mk.pow(2).sum(1).unsqueeze(2)
    #[B,N,C]->[B,C,1]
    #print('a_sq,[B,N,C]->[B,C,1]',a_sq.shape)
        
    two_ab = 2 * (mk.transpose(1, 2) @ qk)
    #[B,C,N]*[B,N2,C] -> [B,N2,N1]
    #print(mk.transpose(1, 2).shape)
    #print(qk.shape)
    #print('two_ab,[B,C,N]*[B,N2,C] -> [B,N2,N1]',two_ab.shape)
        
    similarity = (-a_sq+two_ab)#||^2 distance

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)   # B*N*HW

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=2)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=2, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(2, indices, x_exp) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(2, indices, x_exp) # B*N*HW
    else:
        maxes = torch.max(similarity, dim=2, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=2, keepdim=True)
        affinity = x_exp / x_exp_sum 
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(mk, ms, qk, qe):
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv):
    
    mem = torch.bmm(affinity,mv)
    
    return mem



#cross attention module for the memory mechanism
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # 确保 query, key 和 value 的维度能够被头数整除
        assert query_dim % num_heads == 0
        assert key_dim % num_heads == 0
        assert value_dim % num_heads == 0

        self.q_depth = query_dim // self.num_heads
        self.v_depth = value_dim // self.num_heads

        self.query_linear = nn.Linear(query_dim, query_dim)
        self.key_linear = nn.Linear(key_dim, query_dim)  # 注意这里把key也映射到了query_dim
        self.value_linear = nn.Linear(value_dim, value_dim)

    def split_heads(self, x, batch_size):
        # 分割最后一个维度到 (num_heads, depth)
        dim = x.shape[-1]
        depth = dim // self.num_heads
        x = x.view(batch_size, -1, self.num_heads, depth)
        # 转置以得到维度 (batch_size, num_heads, seq_length, depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # 通过全连接层并分割头
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        # 计算得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.q_depth)
        scores -= torch.max(scores,dim=-1)[0].unsqueeze(-1) 
        attn = F.softmax(scores, dim=-1)

        # 将注意力权重应用到value上
        #print(attn.shape,value.shape)
        context = torch.matmul(attn, value)

        # 把多头的结果拼接 
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.v_depth)

        return context,attn
