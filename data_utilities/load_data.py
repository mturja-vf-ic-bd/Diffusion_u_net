from data_utilities.data_paths import DataPath

import os
import numpy as np
import json
from datetime import date, datetime

from data_utilities.preprocessing import threshold_network

DX_to_label = {"CN": 1, "SMC": 2, "EMCI": 3, "LMCI": 4, "AD": 5}
def load_structural_data():
    net_dict = {}
    for file in os.listdir(DataPath.NETWORK_DATA):
        net = np.loadtxt(os.path.join(DataPath.NETWORK_DATA, file), dtype=int)
        net_dict[file.split("_")[0]] = net
    return net_dict


def load_pet_data():
    with open(DataPath.PET_DATA) as f:
        pet_data = json.load(f)

    for k, v in pet_data.items():
        keys = ["amyloid", "fdg"]
        feat = np.array([v[feat] for feat in keys])
        pet_data[k] = feat
    return pet_data


def create_temporal_data():
    pet_data = load_pet_data()
    struct_data = load_structural_data()
    with open(DataPath.TEMPORAL_MAPPING_FILE) as f:
        temp_file = json.load(f)

    temp_dataset = {}
    for k, v in temp_file.items():
        pet_feature = []
        struct_feature = []
        for scan in v:
            if scan["network_id"] in pet_data.keys() and scan["network_id"] in struct_data.keys():
                pet_feature.append(pet_data[scan["network_id"]])
                struct_feature.append(struct_data[scan["network_id"]])
        if len(pet_feature) > 0:
            pet_feature = np.array(pet_feature)
            struct_feature = np.array(struct_feature)
            temp_dataset[k] = {"node_feature": pet_feature,
                               "network": struct_feature,
                               "label": [v[i]["dx_data"] for i in range(len(v))]}
    return temp_dataset


def load_amyloid_data():
    import pandas as pd
    amy_data = pd.read_excel(DataPath.AMYLOID_PATH, header=0)
    amy_data.sort_values(["PTID", "EXAMDATE"], inplace=True)
    return amy_data


def load_pheno_data():
    import pandas as pd
    pheno_data = pd.read_excel(DataPath.PHENO_DATA_PATH, header=0)
    pheno_data.sort_values(["PTID", "EXAMDATE"], inplace=True)
    return pheno_data


def min_max_normalize(df):
    return (df - df.min()) /(df.max() - df.min())


def load_full_amy_data_w_avg_net(q=80):
    amy_data = load_amyloid_data()
    pheno_data = load_pheno_data()
    Node_col = [col for col in amy_data.columns if "Node" in col]
    sub_ids = amy_data.PTID.unique()
    features = {}
    for col in Node_col:
        amy_data[col] = min_max_normalize(amy_data[col])
    amy_data = amy_data.replace({"PTGENDER": {"Female": 0, "Male": 1}})
    amy_data["PTEDUCAT"] = min_max_normalize(amy_data["PTEDUCAT"])
    amy_data["AGE"] = min_max_normalize(amy_data["AGE"])
    for ptid in sub_ids:
        dates = list(amy_data[amy_data["PTID"] == ptid]["SCAN"].apply(lambda x: x.split("_")[1]))
        dates = [datetime.strptime(x, '%Y-%m-%d') for x in dates]
        delta = [(dates[i] - dates[0]).days / 3000.0 for i in range(0, len(dates))]
        apoe4 = pheno_data[pheno_data["PTID"] == ptid]["APOE4"].to_numpy()[0]
        features[ptid] = {
            "amy": amy_data[amy_data["PTID"] == ptid][Node_col].to_numpy(),
            "DX": amy_data[amy_data["PTID"] == ptid]["DX"].to_numpy(),
            "AGE": amy_data[amy_data["PTID"] == ptid]["AGE"].to_numpy(),
            "GENDER": amy_data[amy_data["PTID"] == ptid]["PTGENDER"].to_numpy(),
            "EDU": amy_data[amy_data["PTID"] == ptid]["PTEDUCAT"].to_numpy(),
            "time": delta,
            "apoe4": apoe4
        }

    # Average network
    net_data = load_structural_data()
    with open(DataPath.TEMPORAL_MAPPING_FILE) as f:
        temp_file = json.load(f)

    dx_to_scan = {'1':[], '2':[], '3':[], '4':[]}
    dx_map = {'1': 'CN', '2':'EMCI', '3':'LMCI', '4':'AD'}
    for k, v in temp_file.items():
        for item in v:
            dx_to_scan[item['dx_data']].append(item['network_id'])
    dx_to_net = {'CN':[], 'EMCI':[], 'LMCI':[], 'AD':[]}
    for k, v in dx_to_scan.items():
        for scan in v:
            if scan in net_data.keys():
                dx_to_net[dx_map[k]].append(net_data[scan])
    for k, v in dx_to_net.items():
        dx_to_net[k] = np.stack(dx_to_net[k], axis=0).mean(axis=0)
        dx_to_net[k] = (dx_to_net[k] + dx_to_net[k].T) / 2
        dx_to_net[k] = threshold_network(dx_to_net[k], deg_norm=True, q=q)
    return features, dx_to_net


def create_temporal_amyloid_data(normalize_feat=True):
    amy_data = load_amyloid_data()
    net_data = create_temporal_data()
    data = {}
    for k, v in net_data.items():
        network = v["network"].mean(axis=0)
        Node_col = [col for col in amy_data.columns if "Node" in col]
        features = amy_data[amy_data["PTID"] == k][Node_col].to_numpy()
        fdg = v["node_feature"][0, 0, :]
        if normalize_feat:
            for col in Node_col:
                amy_data[col] = min_max_normalize(amy_data[col])
        age = list(amy_data[amy_data["PTID"] == k]["AGE"])[0]
        sex = list(amy_data[amy_data["PTID"] == k]["PTGENDER"])[0]
        sex = 0 if sex == "Female" else 1
        dates = list(amy_data[amy_data["PTID"] == k]["SCAN"].apply(lambda x: x.split("_")[1]))
        dates = [datetime.strptime(x, '%Y-%m-%d') for x in dates]
        delta = [(dates[i] - dates[0]).days / 3000.0 for i in range(0, len(dates))]
        amy_delta = (features[1:] - features[:-1]) / (np.array(delta[1:]) - np.array(delta[:-1])).reshape(-1, 1)
        data[k] = {"network": network,
                   "amy": features,
                   "fdg": fdg,
                   "age": age,
                   "time": delta,
                   "dx": int(v["label"][0]),
                   "amy_delta" : amy_delta,
                   "sex": sex}

    return data


if __name__ == '__main__':
    data = load_full_amy_data_w_avg_net()
    data = create_temporal_amyloid_data()
    count_dict = np.zeros((4, 6))
    label_dict = {0:"Normal", 1:"EMCI", 2:"LMCI", 3:"AD"}
    for k, v in data.items():
        count_dict[v["dx"] - 1, v["amy"].shape[0]] += 1

    import matplotlib.pylab as pylab

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (25, 10),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 20,
              'ytick.labelsize': 20}
    pylab.rcParams.update(params)
    for i in range(len(count_dict)):
        pylab.subplot(1, 4, i + 1)
        pylab.xlabel("Number of time points", fontsize=20)
        pylab.ylabel("Number of subjects", fontsize=20)
        pylab.ylim(0, 20)
        pylab.title(label_dict[i], fontsize=25)
        pylab.bar(np.arange(0, 6), count_dict[i], align='center')
    pylab.savefig("data_distribution")
    pylab.show()



