# Min-Max Classifier



import torch
import torch.nn as nn
from torch.distributions import Categorical

from layers import UNet
from layers.GCN import GCNClassifier
from reconstruction_models.adaptive_u_net import HeatDiffusionKernel


class MinMaxClassifier(nn.Module):
    def __init__(self, h_model, n_class=2, act=nn.ELU(), drop_p=0.1, k=2, cls_hdn=256, cls_layer_count=3):
        super(HeatDiffClassifier, self).__init__()
        self.heat_kernel = h_model
        self.gcn_cls = GCNClassifier(ks=2, in_feat=2, dim=16, act=act, k=k, drop_p=drop_p, nodes=148,
                                     cls_hidden=cls_hdn, cls_layer_count=cls_layer_count, n_class=n_class)

    def forward(self, phi_0, g, t, U, V, l):
        predictions, beta, x_0, diff_norm = self.heat_kernel(phi_0, g, t, U, V, l)
        min_max_prog = torch.stack([torch.min(predictions, dim=1)[0], torch.max(predictions, dim=1)[0]], dim=-1)
        final_cls_score = self.gcn_cls(min_max_prog, g)
        return predictions.squeeze(), final_cls_score, beta, x_0, diff_norm
