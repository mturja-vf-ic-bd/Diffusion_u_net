import torch.nn as nn

from layers.layers import SimpleGCN


def generate_layers(dropout, dim, cls_hidden, act, cls_layer_count, n_class, n_nodes):
    drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    cls_hidden_layers = [
        nn.Sequential(
            drop,
            nn.Linear(n_nodes * dim, cls_hidden),
            act
        )
    ]
    cls_hidden_layers.extend(
        [
            nn.Sequential(
                drop,
                nn.Linear(cls_hidden, cls_hidden),
                act
            ) for _ in range(cls_layer_count)
        ]
    )
    cls_hidden_layers.append(
        nn.Linear(cls_hidden, n_class)
    )

    return cls_hidden_layers


class GCNClassifier(nn.Module):
    def __init__(self, ks, in_feat, dim, act, k, drop_p, nodes, cls_hidden, cls_layer_count, n_class):
        super(GCNClassifier, self).__init__()
        self.gcns = nn.ModuleList()
        self.ks = ks
        self.drop = nn.Dropout(drop_p)
        for i in range(ks):
            self.gcns.append(SimpleGCN(dim, dim, act, drop_p, k))
        self.input_proj = nn.Sequential(
            self.drop,
            nn.Linear(in_feat, dim),
            act
        )
        cls_layers = generate_layers(
            dropout=drop_p,
            dim=dim,
            act=act,
            cls_layer_count=cls_layer_count,
            n_nodes=nodes,
            n_class=n_class,
            cls_hidden=cls_hidden
        )
        self.dense = nn.Sequential(*cls_layers)

    def forward(self, x, g):
        x = self.input_proj(x)
        for i in range(self.ks):
            x = self.gcns[i](g, x)
        x = self.dense(x.flatten(start_dim=1))
        return x