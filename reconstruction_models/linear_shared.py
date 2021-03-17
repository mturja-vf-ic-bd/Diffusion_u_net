# We found beta doesn't matter that much and Things aren't that different across groups
# in terms of propagation speed. May be a little higher in AD though
# In this model, We want to learn an embedding x_0 with l1 constraint and few other tweaks.


import torch
import torch.nn as nn


class HeatDiffusionKernel(nn.Module):
    def __init__(self, in_feat, dim, n_modes, ks, act=nn.ELU(), drop_p=0.1, k=2, device="cpu"):
        super(HeatDiffusionKernel, self).__init__()
        self.n_modes = n_modes
        self.device = device

        # init_depo is a neural net to predict x_0 from \phi_0.
        # This might be replaced by any neural net
        self.m = nn.Parameter(torch.FloatTensor(1, 1, 148))
        nn.init.xavier_uniform_(self.m)

    def forward(self, phi_0, t):
        predictions = phi_0.transpose(1, 2) + t.unsqueeze(2)*self.m
        return predictions
