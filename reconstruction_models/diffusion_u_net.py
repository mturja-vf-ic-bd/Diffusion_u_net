# We found beta doesn't matter that much and Things aren't that different across groups
# in terms of propagation speed. May be a little higher in AD though
# In this model, We want to learn an embedding x_0 with l1 constraint and few other tweaks.


import torch
import torch.nn as nn

from layers import UNet


class HeatDiffusionKernel(nn.Module):
    def __init__(self, in_feat, dim, n_modes, ks, act=nn.ELU(), drop_p=0.1, k=2, device="cpu"):
        super(HeatDiffusionKernel, self).__init__()
        self.n_modes = n_modes
        self.device = device
        self.drop = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()
        self.beta = nn.Sequential(
            self.drop,
            nn.Linear(148 * in_feat, 256),
            act,
            self.drop,
            nn.Linear(256, 256),
            act,
            self.drop,
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # init_depo is a neural net to predict x_0 from \phi_0.
        # This might be replaced by any neural net
        self.in_proj = nn.Sequential(
            self.drop,
            nn.Linear(in_feat, dim),
            act,
        )
        self.init_depo = UNet.GraphUnet(ks=ks, dim=dim, act=act, drop_p=drop_p, k=k)
        self.out_proj = nn.Sequential(
            act,
            nn.Linear(dim, 1),
        )

    def generate_gs(self, x_0, t, U, V, beta):
        V = V[0:self.n_modes]
        U = U[:, 0:self.n_modes]
        V_filt = V.unsqueeze(0)
        res = torch.matmul(U, torch.matmul(U.t(), x_0).unsqueeze(0) * (1 - torch.exp(-beta.unsqueeze(0) * V_filt * t.t().unsqueeze(-1))).unsqueeze(-1) / (beta * V_filt).unsqueeze(-1).unsqueeze(0))
        return res

    def forward(self, phi_0, g, t, U, V, l):
        x_0 = self.in_proj(phi_0)
        x_0 = self.init_depo(g, x_0)
        x_0 = self.out_proj(x_0[0][-1])
        beta = self.beta(phi_0.flatten(start_dim=1))
        DF = torch.inverse(torch.diag_embed(torch.ones(beta.shape[0], l.shape[0])).to(self.device) + beta.unsqueeze(-1) * l.unsqueeze(0))
        diff_norm = torch.norm(torch.matmul(DF, x_0), dim=-2)
        predictions = self.generate_gs(x_0, t, U, V, beta) + phi_0[:, :, 0:1]
        return predictions.transpose(0, 1).squeeze(), beta, x_0, diff_norm
