# Training reconstruction with temporal output with v5. This is almost the same as v1_trainer except the
# dynamic part

import torch
import torch.nn as nn

# Prepare paths
import os
import pickle
import time

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.metrics import RunningAverage
from torch.optim.lr_scheduler import MultiStepLR
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from data_utilities.data_paths import DataPath
from reconstruction_models.cluster_u_net import ClusterUnet
from reconstruction_trainer.load_ckp import load_checkpoint
from utils.argument_parser import get_argparser
# Prepare paths
from data_utilities.amy_dataset import Datasetv5
from data_utilities.prepare_data import prepare_data_qced
from utils.custom_metric import ROC_AUC_Multi
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
print("Labels: {}".format(label_all))
# Prepare data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

x, y, dx, g, T = prepare_data_qced(label_all)
x[:, :, 5] = torch.where(x[:, :, 5] > 0, torch.ones_like(x[:, :, 5]), torch.zeros_like(x[:, :, 5]))
x = x.to(device)
print(x.shape)
y = y.to(device)
dx = dx.to(device)
g = g.to(device)
T = T.to(device)
total = dx.shape[0]
train_split = int(total * args.train_r)
valid_split = int(total * args.valid_r)
init_dataset = Datasetv5(x=x, y=y, z=dx, a=g, t=T)
lengths = [train_split, valid_split, len(init_dataset) - train_split - valid_split]
print("Split length: ", lengths)
train, test, eval = random_split(init_dataset, lengths, generator=torch.Generator().manual_seed(4))

train_loader = DataLoader(train, batch_size=args.batch_size)
valid_loader = DataLoader(test, batch_size=valid_split)
eval_loader = DataLoader(eval, batch_size=len(init_dataset) - train_split - valid_split)
print("Data Loaded !!!")
print("Data Loaded !!!")

# Prepare model
dropout = args.dropout
act = nn.ELU()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


with open(DataPath.PREFIX + "/Diffusion_u_net/data_utilities/avg_data.p", "rb") as f:
    data = pickle.load(f)

load_model = args.load
train_model = args.train
model_name = "saved_model"
cn_net = torch.FloatTensor(data["net"]["CN"]).to(device)
cn_net = (cn_net + cn_net.t()) * 0.5
l = torch.diag_embed(cn_net.sum(dim=-1)) - cn_net
V, U = torch.symeig(l, eigenvectors=True)
recon_mdir = os.path.join(FILE_PREFIX, "Diffusion_u_net/log", args.prev_model, "model/saved_model")
with open(recon_mdir, "rb") as f:
    r_model = torch.load(f, map_location=torch.device(device)).to(device)

if load_model:
    if args.model_ckp_path == "saved_model":
        with open(os.path.join(model_log_dir, model_name), "rb") as f:
            model = torch.load(f, map_location=torch.device(device)).to(device)
    else:
        model = ClusterUnet(h_model=r_model, act=nn.ELU(), k=args.n_neigh, drop_p=args.dropout).to(device)
        ckp = load_checkpoint(model, os.path.join(model_log_dir, args.model_ckp_path))
        model.load_state_dict(ckp["model"])
        model = model.to(device)
        model.device = device
else:
    model = ClusterUnet(h_model=r_model, act=nn.ELU(), k=args.n_neigh, drop_p=args.dropout).to(device)
    model.apply(init_weights)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.opt == "Adam" else torch.optim.SGD(model.parameters(), lr=args.lr)
criterion_recon = torch.nn.MSELoss()
criterion_cls = torch.nn.CrossEntropyLoss()
step_scheduler = MultiStepLR(optimizer, milestones=[args.split], gamma=0.1)
lr_scheduler = LRScheduler(step_scheduler)


def compute_loss(y_pred, y, x, dx_pred, dx, x_0, diff_norm, sex_score, apoe_score):
    dx = dx.squeeze().long()
    loss_temp = torch.mul((y_pred - y) ** 2, (y != 0)).sum(dim=1).mean() * args.loss_w[0]
    loss_cls = criterion_cls(dx_pred, dx) * args.loss_w[1]
    loss_ent = -(torch.softmax(x_0, dim=1) * torch.log_softmax(x_0, dim=1)).sum(dim=1).mean() * args.loss_w[2]
    loss_abs = torch.abs(diff_norm).mean() * args.loss_w[3]
    loss_sex = criterion_cls(sex_score, x[:, 0, 2].long()) * args.loss_w[4]
    loss_apoe = criterion_cls(apoe_score, x[:, 0, 5].long()) * args.loss_w[5]
    return loss_temp + loss_cls + loss_ent + loss_abs + loss_sex + loss_apoe, \
           loss_temp, loss_cls, loss_ent, loss_abs, loss_sex, loss_apoe


# Prepare trainer
def process_function(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y, dx, g, t = batch
    y_pred, final_cls_score, beta, x_0, diff_norm, sex_score, apoe_score = model(x[:, :, 0:2], g, t, U, V, l)
    loss, recon_loss, cls_loss, ent_loss, l1_loss, loss_sex, loss_apoe = compute_loss(y_pred,
                                                                     y, x, final_cls_score,
                                                                     dx, x_0, diff_norm,
                                                                     sex_score, apoe_score)
    loss.backward()
    optimizer.step()
    return {"prediction": y_pred, "true": y, "loss": loss.item(), "dx_pred": final_cls_score, "dx": dx,
                "recon_loss": recon_loss.item(), "cls_loss": cls_loss.item(), "ent_loss": ent_loss.item(),
                "l1_loss": l1_loss.item(), "sex_loss": loss_sex, "loss_apoe": loss_apoe}


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y, dx, g, t = batch
        y_pred, final_cls_score, beta, x_0, diff_norm, sex_score, apoe_score = model(x[:, :, 0:2], g, t, U, V, l)
        loss, recon_loss, cls_loss, ent_loss, l1_loss, loss_sex, loss_apoe = compute_loss(y_pred,
                                                                     y, x, final_cls_score,
                                                                     dx, x_0, diff_norm,
                                                                     sex_score, apoe_score)
        return {"prediction": y_pred, "true": y, "loss": loss.item(), "dx_pred": final_cls_score, "dx": dx,
                "recon_loss": recon_loss.item(), "cls_loss": cls_loss.item(), "ent_loss": ent_loss.item(),
                "l1_loss": l1_loss.item(), "sex_loss": loss_sex, "loss_apoe": loss_apoe}


def print_logs(engine, dataloader, mode, history_dict):
    metrics = ["loss", "recon_loss", "ent_loss", "l1_loss", "cls_loss", "sex_loss", "loss_apoe"]
    evaluator.run(dataloader, max_epochs=1)
    print_line = mode + ": " + str(engine.state.epoch) + ": " + ", ROC: "
    for i in range(len(set(label_all.values()))):
        print_line += "{:.4f}, ".format(evaluator.state.metrics[i])
        tb_log.add_scalar("{}/{}_ROC".format(mode, i), evaluator.state.metrics[i], engine.state.epoch)
    for m in metrics:
        v = evaluator.state.output[m]
        tb_log.add_scalar("{}/{}".format(mode, m), v, engine.state.epoch)
        print_line += "{}: {:.4f}, ".format(m, v)
        if m not in history_dict.keys():
            history_dict[m] = [v]
        else:
            history_dict[m].append(v)
    print(print_line)


def activated_output_transform(output):
    y_pred, y = output["dx_pred"], output["dx"]
    y_pred = torch.sigmoid(y_pred)
    return y, y_pred


if train_model:
    trainer = Engine(process_function)
    evaluator = Engine(evaluate_function)
    roc_auc = ROC_AUC_Multi(len(set(label_all.values())), activated_output_transform)
    roc_auc.attach(evaluator, "ROC")
    roc_auc.attach(trainer, "ROC")
    metric_names = ["loss", "recon_loss", "ent_loss", "l1_loss", "cls_loss"]
    training_history = {}
    validation_history = {}
    evaluation_history = {}
    for m in metric_names:
        RunningAverage(output_transform=lambda x: x[m]).attach(trainer, m)
        RunningAverage(output_transform=lambda x: x[m]).attach(evaluator, m)
    to_save = {
        'model': model,
        'optimizer': optimizer,
        'trainer': trainer
    }

    def score_function(engine):
        val_loss = evaluator.state.metrics['micro']
        return val_loss

    handler = Checkpoint(to_save,
                         DiskSaver(model_log_dir, create_dir=True, require_empty=False),
                         score_name="val_loss",
                         filename_prefix='min_max_best',
                         score_function=score_function,
                         global_step_transform=global_step_from_engine(trainer),
                         n_saved=10)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, train_loader, 'Training', training_history)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, valid_loader, 'Validation', validation_history)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), handler)
    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    start_time = time.time()
    trainer.run(train_loader, max_epochs=args.max_epoch)
    with open(os.path.join(model_log_dir, model_name), "wb") as f:
        torch.save(model, f)
    elapsed = time.time() - start_time
    print("Time elapsed: {}h {}m".format(elapsed // 3600, (elapsed % 3600) // 60))
