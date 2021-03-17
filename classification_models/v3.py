# GRU + GCNClassifier

import torch
import torch.nn as nn
from layers.GCN import GCNClassifier


class HeatDiffClassifier(nn.Module):
    def __init__(self, h_model, n_class=2, dim=64, act=nn.ELU(), drop_p=0.1, k=2, cls_hdn=256, cls_layer_count=3):
        super(HeatDiffClassifier, self).__init__()
        self.heat_kernel = h_model
        self.gru = nn.GRU(input_size=148, hidden_size=148, dropout=drop_p, batch_first=True)
        self.gcn_cls = GCNClassifier(ks=2, in_feat=1, dim=dim, act=act, k=k, drop_p=drop_p, nodes=148,
                                     cls_hidden=cls_hdn, cls_layer_count=cls_layer_count, n_class=n_class)

    def forward(self, phi_0, g, t, U, V, l):
        predictions, beta, x_0, diff_norm = self.heat_kernel(phi_0, g, t, U, V, l)
        x, _ = self.gru(torch.cat([phi_0[:, :, 0].unsqueeze(1), predictions], dim=1))
        x = x.sum(dim=1)
        final_cls_score = self.gcn_cls(x.unsqueeze(-1), g)
        return final_cls_score
