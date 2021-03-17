# Training reconstruction with temporal output with v5. This is almost the same as v1_trainer except the
# dynamic part

import torch

# Prepare paths
import os

from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utilities.data_paths import DataPath
from utils.argument_parser import get_argparser
# Prepare paths
from data_utilities.amy_dataset import Datasetv5
from data_utilities.prepare_data import prepare_data_qced
import json


parser = get_argparser()
args = parser.parse_args()
print(args)


FILE_PREFIX = DataPath.PREFIX
folder_name = args.write_dir
log_dir = os.path.join(FILE_PREFIX, "Diffusion_u_net/log/" + folder_name)
tb_log_dir = log_dir + "/tb"
model_log_dir = log_dir + "/model"
result_log_dir = log_dir + "/result"
tb_log = SummaryWriter(tb_log_dir)
label_all = json.load(open("classes.json"))
with open(os.path.join(log_dir, "Arguments"), "w") as f:
    f.write(str(args))
    f.write("Labels: {}".format(label_all))
# Prepare data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

x, y, dx, g, T = prepare_data_qced(label_all)
total = dx.shape[0]
train_split = int(total * args.train_r)
valid_split = int(total * args.valid_r)
init_dataset = Datasetv5(x=x, y=y, z=dx, a=g, t=T)
lengths = [train_split, valid_split, len(init_dataset) - train_split - valid_split]
print("Split length: ", lengths)
train, test, eval = random_split(init_dataset, lengths, generator=torch.Generator().manual_seed(4))
x = x[:, :, 0].numpy()
dx = dx[:, 0].numpy()


# Define model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
kf = KFold(n_splits=10, shuffle=True, random_state=4)
clf = RandomForestClassifier(n_estimators=100, max_depth=32, random_state=0, min_samples_split=12, max_features='sqrt', bootstrap=True)

roc_scores = np.zeros(10)
i = 0
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = dx[train_index], dx[test_index]
    clf.fit(X_train, y_train)
    rf_probs = clf.predict_proba(X_test)[:, 1]
    roc_value = roc_auc_score(y_test, rf_probs)
    roc_scores[i] = roc_value
    i = i + 1
print(roc_scores)
print(roc_scores.mean())