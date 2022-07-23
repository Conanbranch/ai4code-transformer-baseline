import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import sys, os
from metrics import *
import torch
import argparse
import shutil

parser = argparse.ArgumentParser(description='process arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', help='path for pretrained model')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv', help='path for markdown validation data')
parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json', help='path for code validation data')
parser.add_argument('--val_path', type=str, default="./data/val.csv", help='path for validation data')
parser.add_argument('--model_ckp_path', type=str, default="./output", help='path for model and model checkpoints')
parser.add_argument('--model_ckp', type=str, default="model.pt", help='model checkpoint filename')
parser.add_argument('--model', type=str, default="model.bin", help='model filename')

args = parser.parse_args()
    
if not os.path.exists("./output"):
    os.mkdir("./output")      
      
data_dir = Path('..//input/')

val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
#val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

#val_df_mark['source'] = val_df_mark['source'].fillna('')
#val_df['source'] = val_df['source'].fillna('')

order_df = pd.read_csv("../input/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts, code_sep_token = args.code_sep_token, 
                         pad_between_code = args.pad_between_code, vbl_code=args.vbl_code)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)
    
    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
    
    return np.concatenate(labels), np.concatenate(preds)

def predict(model_path, ckpt_path):
    model = MarkdownModel(model_path, re_init = True, reinit_n_layers = 1)
    model = model.cuda()
    model.eval()
    model.load_state_dict(torch.load(ckpt_path))
    y_val, y_pred = validate(model, val_loader)
    return y_val, y_pred

def eval(y_val, y_pred):
    #val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    val_df["pred"] = val_df["pct_rank"]
    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
    y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
    score = print("pred score", score)
    return score

num_models = 2

model_path = "../input/codebert-base/codebert-base/"
ckpt_path = "../input/a14codemodels/epoch_5_model_current_new_r_f_40_init_1_sigmoid_e_10.bin" 
y_val, y_test_1 = predict(model_path, ckpt_path)

model_path = "../input/codebert-base/codebert-base/"
ckpt_path = "../input/a14codemodels/epoch_5_model_current_new_r_f_40_init_1_sigmoid_e_10.bin" 
y_val, y_test_2 = predict(model_path, ckpt_path)

# define weights to consider
w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
best_score, best_weights = 0.0, None
# iterate all possible combinations (cartesian product)
for weights in product(w, repeat=num_models):
    y_pred = ((y_test_1*weights[0]) + (y_test_2*weights[1]))/np.sum(weights)
    score = eval(y_val, y_pred)
    if score > best_score:
        best_score, best_weights = score, weights
        print('>%s %.3f' % (best_weights, best_score))
