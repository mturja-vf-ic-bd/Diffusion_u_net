from scipy.sparse.csgraph import laplacian
from sklearn.linear_model import LinearRegression

from data_utilities.load_data import create_temporal_amyloid_data
import torch
from data_utilities.preprocessing import power_g, threshold_network
import numpy as np


def convert_to_1hot(a):
    if a == 1:
        return [0, 0, 0, 1]
    elif a == 2:
        return [0, 0, 1, 0]
    elif a == 3:
        return [0, 1, 0, 0]
    elif a == 4:
        return [1, 0, 0, 0]


def prepare_data_sde(device):
    data = create_temporal_amyloid_data(normalize_feat=False)
    node_id = 0
    x = []
    y = []
    t = []
    for k, v in data.items():
        if v["amy"].shape[0] > 1:
            feat = torch.Tensor([v["amy"][0, node_id], v["age"], v["sex"]])
            dx = torch.Tensor(convert_to_1hot(v["dx"]))
            feat = torch.cat((feat, dx))
            if len(v["time"]) < 5:
                for p in range(len(v["time"]), 5):
                    v["time"].append(v["time"][-1] + 0.25)
            x.append(feat)
            if v["amy"].shape[0] < 5:
                o = torch.Tensor(np.concatenate((v["amy"][1:, node_id], -1*np.ones((5 - v["amy"].shape[0], )))))
            else:
                o = torch.Tensor(v["amy"][1:, node_id])
            y.append(o)
            t.append(torch.Tensor(v["time"]))

    x = torch.stack(x, dim=0).to(device)
    for ind in range(2):
        x[:, ind] -= x[:, ind].mean(dim=0)
        x[:, ind] /= x[:, ind].std(dim=0)

    y = torch.stack(y, dim=0).to(device)
    return x, y, torch.stack(t, dim=0).to(device)


def prepare_data_sde_all_node(device):
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    y = []
    max_t = 6
    for k, v in data.items():
        if v["amy"].shape[0] > 1:
            if v["amy"].shape[0] < max_t:
                inp = torch.Tensor(np.concatenate((v["amy"][0:-1, :], -1*np.ones((max_t - v["amy"].shape[0], v["amy"].shape[1])))))
                o = torch.Tensor(np.concatenate((v["amy"][1:, :], -1*np.ones((max_t - v["amy"].shape[0], v["amy"].shape[1])))))
            else:
                inp = torch.Tensor(v["amy"][0:max_t-1, :])
                o = torch.Tensor(v["amy"][1:max_t, :])
            x.append(inp)
            y.append(o)

    x = torch.stack(x, dim=0).to(device)
    y = torch.stack(y, dim=0).to(device)
    return x, y


def prepare_data_mle():
    data = create_temporal_amyloid_data()
    node_id = 0
    x = []
    y = []
    t = []
    for k, v in data.items():
        if v["amy"].shape[0] > 1:
            for j in range(v["amy"].shape[0] - 1):
                net = (v["network"] + v["network"].T) // 2
                diff = np.matmul(laplacian(net, normed=True), v["amy"][j].reshape(-1, 1))
                feat = torch.Tensor([v["amy"][j, node_id], v["age"], v["sex"], v["time"][j+1]])
                dx = torch.Tensor(convert_to_1hot(v["dx"]))
                feat = torch.cat((feat, dx))
                x.append(feat)
            y += list(v["amy"][1:, node_id])

    x = torch.stack(x, dim=0)
    y = torch.Tensor(y).view(-1, 1)
    return x, y


def prepare_data_attn():
    data = create_temporal_amyloid_data()
    node_id = 0
    x = []
    y = []
    net = []
    for k, v in data.items():
        if v["dx"] != 2:
            for j in range(v["amy"].shape[0] - 1):
                feat = torch.Tensor([v["age"], v["sex"]]).view(-1, 1).repeat((1, v["amy"].shape[1]))
                amy = torch.Tensor(v["amy"][j, :]).unsqueeze(0)
                feat = torch.cat([amy, feat], dim=0)
                x.append(feat.transpose(0, 1))
                net.append(v["network"])
                if v["dx"] == 1:
                    y.append(0)
                else:
                    y.append(1)

    x = torch.stack(x, dim=0)
    x[:, :, 1] -= x[:, :, 1].mean()
    x[:, :, 1] /= x[:, :, 1].std()
    y = torch.Tensor(y).view(-1, 1)
    net = torch.Tensor(net)
    return x, y, net


def prepare_data_baseline_amy(label_map):
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    y = []
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            x.append(v["amy"][0, :])
            y.append(label_map[v["DX"][0]])

    x = torch.Tensor(x)
    y = torch.Tensor(y).unsqueeze(1)
    return x, y


def prepare_data_amy():
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data = load_full_amy_data_w_avg_net()
    x = []
    y = []
    label_map = {"CN": 0, "SMC": 0, "LMCI": 1, "AD": 1}
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            for j in range(len(v["amy"])):
                x.append(v["amy"][j, :])
                y.append(label_map[v["DX"][0]])
    x = torch.Tensor(x)
    y = torch.Tensor(y).unsqueeze(1)
    return x, y


def prepare_data_amy_with_temp_slope():
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    dx = []
    label_map = {"CN": 0, "SMC": 0, "LMCI": 1, "AD": 1}
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            amy = v["amy"]
            # amy = v["amy"] / v["amy"].mean(axis=1)[:, np.newaxis]
            slope = np.zeros(v["amy"].shape[1])
            if v["amy"].shape[0] > 1:
                for i in range(v["amy"].shape[1]):
                    reg = LinearRegression().fit(np.arange(v["amy"].shape[0])[:, np.newaxis], amy[:, i])
                    slope[i] = reg.coef_[0]
            x.append(np.concatenate((amy[0, :], slope)))
            dx.append(label_map[v["DX"][0]])

    x = torch.Tensor(x)
    dx = torch.Tensor(dx)
    return x, dx.unsqueeze(1)


def prepare_data_amy_with_temp_slope_graph_feat(K=3):
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    dx = []
    g = []
    label_map = {"CN": 0, "SMC": 0, "LMCI": 1, "AD": 1}
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            amy = v["amy"]
            slope = np.zeros(v["amy"].shape[1])
            if v["amy"].shape[0] > 1:
                for i in range(v["amy"].shape[1]):
                    reg = LinearRegression().fit(np.arange(v["amy"].shape[0])[:, np.newaxis], amy[:, i])
                    slope[i] = reg.coef_[0]
            G = net["CN"]
            graph_feats = [amy[0, :], slope]
            for l in range(K):
                graph_feats.append(power_g(G, amy[0, :], l+1))
                graph_feats.append(power_g(G, slope, l+1))
            x.append(np.stack(graph_feats, axis=1))
            dx.append(label_map[v["DX"][0]])
            g.append(G)

    x = torch.Tensor(x)
    dx = torch.Tensor(dx)
    g = torch.Tensor(g)
    return x, dx.unsqueeze(1), g


def prepare_data_amy_with_temp_graph_data(label_map=None):
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    dx = []
    g = []
    y = []
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            amy = v["amy"]
            slope = np.zeros(v["amy"].shape[1])
            if v["amy"].shape[0] > 1:
                for i in range(v["amy"].shape[1]):
                    reg = LinearRegression().fit(np.arange(v["amy"].shape[0])[:, np.newaxis], amy[:, i])
                    slope[i] = reg.coef_[0]
            G = net["CN"]
            graph_feats = [amy[0, :], slope, power_g(G, amy[0, :], 1), power_g(G, amy[0, :], 2), power_g(G, amy[0, :], 3)]
            x.append(np.stack(graph_feats, axis=1))
            dx.append(label_map[v["DX"][0]])
            g.append(G)
            y_temp = np.zeros((5, amy.shape[1]))
            if amy.shape[0] > 1:
                y_temp[0:amy.shape[0]-1, :] = amy[1:, :]
            y.append(y_temp)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    dx = torch.Tensor(dx)
    g = torch.Tensor(g)
    return x, y, dx.unsqueeze(1), g


def prepare_data_v8(label_map=None):
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    dx = []
    g = []
    y = []
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            amy = v["amy"]
            slope = np.zeros(v["amy"].shape[1])
            if v["amy"].shape[0] > 1:
                for i in range(v["amy"].shape[1]):
                    reg = LinearRegression().fit(np.arange(v["amy"].shape[0])[:, np.newaxis], amy[:, i])
                    slope[i] = reg.coef_[0]
            G = net["CN"]
            graph_feats = [amy[0, :], slope, np.ones((slope.shape[0], )) * v["GENDER"][0],
                           np.ones((slope.shape[0], )) * v["AGE"][0],
                           np.ones((slope.shape[0], )) * v["EDU"][0]]
            x.append(np.stack(graph_feats, axis=1))
            dx.append(label_map[v["DX"][0]])
            g.append(G)
            y_temp = np.zeros((5, amy.shape[1]))
            if amy.shape[0] > 1:
                y_temp[0:amy.shape[0] - 1, :] = amy[1:, :]
            y.append(y_temp)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    dx = torch.Tensor(dx)
    g = torch.Tensor(g)
    return x, y, dx.unsqueeze(1), g


def qc_amy(amy):
    if amy.shape[0] == 1:
        return [], 0
    diff_forward = amy[1:] - amy[:-1]
    diff_backward = amy[:-1] - amy[1:]
    count = 0
    idx = [0]
    for i in range(diff_forward.shape[0]):
        if i < diff_forward.shape[0] - 1 and (diff_forward[i] < 0).all() and (diff_backward[i+1] < 0).all():
            count = count + 1
        elif i == diff_forward.shape[0] - 1 and (diff_forward[i] < 0).all():
            count = count + 1
        else:
            idx.append(i+1)

    return idx, count


def prepare_data_qced(label_map):
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    dx = []
    g = []
    y = []
    T = []
    G = net["CN"]
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            amy = v["amy"]
            time = np.array(v["time"])
            idx, c = qc_amy(amy)
            idx = np.array(idx)
            if c > 0:
                amy = amy[idx, :]
                time = time[idx]
            slope = np.zeros(amy.shape[1])
            if amy.shape[0] > 1:
                for i in range(amy.shape[1]):
                    reg = LinearRegression().fit(np.arange(amy.shape[0])[:, np.newaxis], amy[:, i])
                    slope[i] = reg.coef_[0]
            graph_feats = [amy[0, :], slope, np.ones((slope.shape[0],)) * v["GENDER"][0],
                           np.ones((slope.shape[0],)) * v["AGE"][0],
                           np.ones((slope.shape[0],)) * v["EDU"][0],
                           np.ones((slope.shape[0],)) * v["apoe4"]]
            x.append(np.stack(graph_feats, axis=1))
            dx.append(label_map[v["DX"][0]])
            g.append(G)
            t = np.zeros((5, ))
            if time.shape[0] > 0:
                t[0:time.shape[0] - 1] = time[1:]
            T.append(t)
            y_temp = np.zeros((5, amy.shape[1]))
            if amy.shape[0] > 1:
                y_temp[0:amy.shape[0] - 1, :] = amy[1:, :]
            y.append(y_temp)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    dx = torch.Tensor(dx)
    g = torch.Tensor(g)
    T = torch.Tensor(T)
    return x, y, dx.unsqueeze(1), g, T


def prepare_data_v12(label_map):
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    x = []
    dx = []
    g = []
    y = []
    demo = []
    for k, v in data.items():
        if v["DX"][0] in label_map.keys():
            amy = v["amy"]
            slope = np.zeros(v["amy"].shape[1])
            if v["amy"].shape[0] > 1:
                for i in range(v["amy"].shape[1]):
                    reg = LinearRegression().fit(np.arange(v["amy"].shape[0])[:, np.newaxis], amy[:, i])
                    slope[i] = reg.coef_[0]
            G = net["CN"]
            lap = laplacian((G + G.T) / 2, normed=True)
            lap_feat = np.matmul(v["amy"][0], lap)
            graph_feats = [amy[0, :], slope, lap_feat]
            x.append(np.stack(graph_feats, axis=1))
            dx.append(label_map[v["DX"][0]])
            g.append(G)
            y_temp = np.zeros((5, amy.shape[1]))
            if amy.shape[0] > 1:
                y_temp[0:amy.shape[0] - 1, :] = amy[1:, :]
            y.append(y_temp)
            demo.append(np.array([v["GENDER"][0], v["AGE"][0], v["EDU"][0]]))

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    dx = torch.Tensor(dx)
    g = torch.Tensor(g)
    demo = torch.Tensor(demo)
    return x, y, dx.unsqueeze(1), g, demo


def prepare_data_v14_g():
    from data_utilities.load_data import load_full_amy_data_w_avg_net
    data, net = load_full_amy_data_w_avg_net()
    dx_amy = {"CN": [], "SMC": [], "EMCI": [], "LMCI": [], "AD": [], np.nan: []}
    avg_amy = {"CN": [], "SMC": [], "EMCI": [], "LMCI": [], "AD": [], np.nan: []}
    for k, v in data.items():
        amy = v["amy"].mean(axis=0)
        dx_amy[v["DX"][0]].append(amy)
    for dx, v in dx_amy.items():
        v = np.stack(v, axis=-1).mean(axis=-1)
        avg_amy[dx] = v

    import pickle
    with open("avg_data.p", "wb") as f:
        pickle.dump({"avg_amy": avg_amy, "net": net}, f)
    return avg_amy, net


if __name__ == '__main__':
    # data = prepare_data_v8({"SMC": 0, "CN": 0, "LMCI": 1, "AD": 1})
    amy, net = prepare_data_v14_g()
    print(amy)
