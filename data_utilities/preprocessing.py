import numpy as np
import torch


def threshold_network(net, option="percentile", q=80, deg_norm=False):
    if option == "percentile":
        p = np.percentile(net, q=q)
        res = np.where(net > p, net, 0)
    if deg_norm:
        res /= res.sum(axis=1)[:, np.newaxis]
    return res


def power_g(g, x, k=1):
    if k==1:
        return np.matmul(g, x)
    return np.matmul(g, power_g(g, x, k-1))


def get_eigen(g, component=50):
    l = torch.diag_embed(g.sum(dim=-1)) - g
    e, v = torch.symeig(l, eigenvectors=True)
    v_com = v[:, :component]
    return v_com


def spectral_feature(x, v_com):
    return torch.matmul(v_com.t(), x)
