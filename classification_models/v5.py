import torch
import torch.nn as nn

from layers.GCN import GCNClassifier


class JointClassifier(nn.Module):
    def __init__(self, in_feat, dim, n_class=2,
                 act=nn.ELU(), drop_p=0.1, k=2,
                 cls_hdn=256, cls_layer_count=3):
        super(JointClassifier, self).__init__()
        self.gcn_cls = GCNClassifier(ks=2, in_feat=in_feat, dim=dim, act=act, k=k, drop_p=drop_p, nodes=148,
                                     cls_hidden=cls_hdn, cls_layer_count=cls_layer_count, n_class=n_class)

    def forward(self, phi, y, g):
        phi = torch.cat([phi.transpose(1, 2), y], dim=1)
        phi_min = phi.clone()
        phi_min[phi_min == 0] = 100
        inp = torch.stack([torch.min(phi_min, dim=1)[0], torch.max(phi, dim=1)[0]], dim=-1)
        return self.gcn_cls(inp, g)