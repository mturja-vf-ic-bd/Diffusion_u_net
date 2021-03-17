import json
import numpy as np
from scipy.stats import ttest_ind

from data_utilities.data_paths import DataPath
from data_utilities.load_data import load_structural_data
from data_utilities.preprocessing import threshold_network
from sklearn.model_selection import KFold

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
            net = threshold_network(net_data[scan] + net_data[scan].T, deg_norm=False, q=1)
            dx_to_net[dx_map[k]].append(net.flatten())


# kf = KFold(n_splits=2, shuffle=True, random_state=None)
# X = np.array(dx_to_net["EMCI"])
t_th = 0.05
# sum = 0
# n_test = 100
# for i in range(n_test):
#     for a, b in kf.split(X):
#         X_a, X_b = X[a], X[b]
#         t = ttest_ind(X_a, X_b)
#     sum += (t[1] < t_th).sum()
#
# print(sum / n_test)
# print(sum / n_test * 100 / 148 / 148)

print("T Test between CN and EMCI")
t = ttest_ind(dx_to_net["CN"], dx_to_net["AD"])
print((t[1] < t_th).sum())
print((t[1] < t_th).sum() * 100 / 148 / 148)

threshold_network(dx_to_net["CN"], )

