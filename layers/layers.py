# This is an implementation of Graph-Unet motivated from here
# https://github.com/HongyangGao/Graph-U-Nets/blob/b3d2c735877868406673bb7af3dd6b7f7e6d4f02/src/utils/ops.py#L6

import torch
import torch.nn as nn
import numpy as np


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[-1]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)), dim=-1)
    idx_batched = idx.unsqueeze(2).expand(idx.size(0), idx.size(1), h.size(2))
    new_h = torch.gather(h, 1, idx_batched)
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    # un_g = g.bool().float()
    # un_g = torch.matmul(un_g, un_g).bool().float()
    idx_g1 = idx.unsqueeze(2).expand(idx.size(0), idx.size(1), g.size(2))
    un_g = torch.gather(g, 1, idx_g1)
    idx_g2 = idx.unsqueeze(1).expand(idx.size(0), un_g.size(1), idx.size(1))
    un_g = torch.gather(un_g, 2, idx_g2)
    g = norm_g(un_g)
    return g, new_h, idx


def top_k_graph_common(scores, g, h, k):
    num_nodes = g.shape[-1]
    values, idx = torch.topk(scores.mean(dim=0), max(2, int(k*num_nodes)), dim=-1)
    new_h = h[:, idx]
    values = values.unsqueeze(0).unsqueeze(-1)
    new_h = torch.mul(new_h, values)
    g = g[:, idx, :]
    g = g[:, :, idx]
    g = norm_g(g)
    return g, new_h, idx


def norm_g(g):
    EPS =  1e-10
    degrees = torch.sum(g, -1, keepdim=True)
    g = g / (degrees + EPS)
    return g


def diffuse(x, net, k):
    if len(x.shape) == 2:
        f = x.unsqueeze(2)
    else:
        f = x
    diffused_feat = [f]
    for i in range(k):
        f = torch.matmul(net, f)
        diffused_feat.append(f)
    return torch.cat(diffused_feat, dim=-1)


class SimpleGCN(nn.Module):
    def __init__(self, in_feat, out_feat, act, dropout, k):
        super(SimpleGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear((k + 1) * in_feat, out_feat)
        self.act = act
        self.k = k

    def forward(self, net, f):
        f = diffuse(f, net, self.k).flatten(start_dim=2)
        f = self.dropout(f)
        f = self.proj(f)
        f = self.act(f)
        return f


class Pool(nn.Module):
    def __init__(self, k, in_dim, p, common=False):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.common = common
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        if self.common:
            return top_k_graph_common(scores, g, h, self.k)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):
    def __init__(self, common=False, *args):
        super(Unpool, self).__init__()
        self.common = common

    def forward(self, g, h, pre_h, idx):
        new_h = torch.zeros_like(pre_h)
        if self.common:
            new_h[:, idx] = h
        else:
            new_h.scatter_(dim=1, index=idx.unsqueeze(2).expand(idx.shape[0], idx.shape[1], h.shape[2]), src=h)
        return g, new_h


class Initializer(object):
    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
