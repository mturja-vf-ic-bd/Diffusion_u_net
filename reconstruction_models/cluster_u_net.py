import torch
import torch.nn as nn

from layers.GCN import GCNClassifier


class ClusterUnet(nn.Module):
    def __init__(self, h_model, n_class=2, act=nn.ELU(), drop_p=0.1, k=2, cls_hdn=256, cls_layer_count=3):
        super(ClusterUnet, self).__init__()
        self.heat_kernel = h_model
        self.drop = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()
        self.gcn_cls = GCNClassifier(ks=2, in_feat=2, dim=16, act=act, k=k, drop_p=drop_p, nodes=148,
                                     cls_hidden=cls_hdn, cls_layer_count=cls_layer_count, n_class=n_class)
        self.sex_cls = nn.Sequential(
            self.drop,
            nn.Linear(148, 256),
            act,
            self.drop,
            nn.Linear(256, 256),
            act,
            self.drop,
            nn.Linear(256, 256),
            act,
            nn.Linear(256, 2)

        )

        self.apoe_cls = nn.Sequential(
            self.drop,
            nn.Linear(148, 256),
            act,
            self.drop,
            nn.Linear(256, 256),
            act,
            self.drop,
            nn.Linear(256, 256),
            act,
            nn.Linear(256, 2)

        )

    def forward(self, phi_0, g, t, U, V, l):
        predictions, beta, x_0, diff_norm = self.heat_kernel(phi_0, g, t, U, V, l)
        min_max_prog = torch.stack([torch.min(predictions, dim=1)[0], torch.max(predictions, dim=1)[0]], dim=-1)
        sex_score = self.sex_cls(x_0.squeeze())
        apoe_score = self.apoe_cls(x_0.squeeze())
        final_cls_score = self.gcn_cls(min_max_prog, g)
        return predictions.squeeze(), final_cls_score, beta, x_0, diff_norm, sex_score, apoe_score
