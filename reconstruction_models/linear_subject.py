
import torch
import torch.nn as nn


class HeatDiffusionKernel(nn.Module):
    def __init__(self, in_feat, dim, n_modes, ks, n_class, act=nn.ELU(), drop_p=0.1, k=2, device="cpu"):
        super(HeatDiffusionKernel, self).__init__()
        self.n_modes = n_modes
        self.device = device
        self.drop = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()

        # init_depo is a neural net to predict x_0 from \phi_0.
        # This might be replaced by any neural net
        self.in_proj = nn.Sequential(
            nn.Linear(148, 148),
        )

    def forward(self, phi_0, g, t):
        x_0 = self.in_proj(phi_0[:, :, 0])
        predictions = x_0.unsqueeze(-1) * t.unsqueeze(1) + phi_0[:, :, 0:1]
        return predictions.transpose(1, 2)
