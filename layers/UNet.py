import torch.nn as nn
from layers.layers import SimpleGCN, Pool, Unpool


class GraphUnet(nn.Module):
    def __init__(self, ks, dim, act, drop_p, k, common=False):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = SimpleGCN(dim, dim, act, drop_p, k)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(SimpleGCN(dim, dim, act, drop_p, k))
            self.up_gcns.append(SimpleGCN(dim, dim, act, drop_p, k))
            self.pools.append(Pool(ks[i], dim, drop_p, common))
            self.unpools.append(Unpool(dim, dim, drop_p, common))

    def forward(self, g, x, y=None):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = x
        for i in range(self.l_n):
            x = self.down_gcns[i](g, x)
            adj_ms.append(g)
            down_outs.append(x)
            g, x, idx = self.pools[i](g, x)
            indices_list.append(idx)
        x = self.bottom_gcn(g, x)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, x = self.unpools[i](g, x, down_outs[up_idx], idx)
            x = self.up_gcns[i](g, x)
            x = x.add(down_outs[up_idx])
            hs.append(x)
        x = x.add(org_h)
        hs.append(x)
        return hs, indices_list
