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
from itertools import product

parser = argparse.ArgumentParser(description='process arguments')
parser.add_argument('--model_ckp_path', type=str, default="./output", help='path for model and model checkpoints')
parser.add_argument('--model_ckp', type=str, default="model.pt", help='model checkpoint filename')
parser.add_argument('--model_ckp_1', type=str, default="model.pt", help='model checkpoint filename')
parser.add_argument('--model_ckp_2', type=str, default="model.pt", help='model checkpoint filename')
parser.add_argument('--model_ckp_3', type=str, default="model.pt", help='model checkpoint filename')
parser.add_argument('--model_ckp_4', type=str, default="model.pt", help='model checkpoint filename')
parser.add_argument('--model', type=str, default="model.bin", help='model filename')

parser.add_argument('--model_ckp_1_vbl_code', action='store_true', help="use variable length code")
parser.add_argument('--model_ckp_2_vbl_code', action='store_true', help="use variable length code")
parser.add_argument('--model_ckp_3_vbl_code', action='store_true', help="use variable length code")
parser.add_argument('--model_ckp_4_vbl_code', action='store_true', help="use variable length code")

parser.add_argument('--steps', type=int, default=21, help="number of steps for weights")
parser.add_argument('--num_models', type=int, default=2, help="number of steps for weights")

parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', help='path for pretrained model')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv', help='path for markdown validation data')
parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json', help='path for code validation data')
parser.add_argument('--val_path', type=str, default="./data/val.csv", help='path for validation data')
parser.add_argument('--md_max_len', type=int, default=64, help='maximum length of tokenized markdown')
parser.add_argument('--total_max_len', type=int, default=512, help='maximum length of tokenized markdown and code')
parser.add_argument('--batch_size', type=int, default=8, help='training batchsize, try --batch_size 8 if you encounter memory issues')
parser.add_argument('--n_workers', type=int, default=8, help='number of workers')
parser.add_argument('--re_init', action='store_true', help="option to re-initialize layers of the pretrained model")
parser.add_argument('--reinit_n_layers', type=int, default=0, help="number of layers of the pretrained model to re-initialize")
parser.add_argument('--code_sep_token', action='store_true', help="include seperator tokens between code samples")
parser.add_argument('--pad_between_code', action='store_true', help="include seperator tokens between code samples")

args = parser.parse_args()
    
if not os.path.exists("./output"):
    os.mkdir("./output")      
      
data_dir = Path('..//input/')

#val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

val_df_mark['source'] = val_df_mark['source'].fillna('')
val_df['source'] = val_df['source'].fillna('')

order_df = pd.read_csv("../input/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

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

def predict(model_path, ckpt_path, ckpt, vbl_code):
    model = MarkdownModel(model_path, re_init = args.re_init, reinit_n_layers = args.reinit_n_layers)
    model = model.cuda()
    model.eval()
    model.load_state_dict(torch.load(ckpt_path + '/' + ckpt))
    val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts, code_sep_token = args.code_sep_token, 
                         pad_between_code = args.pad_between_code, vbl_code=vbl_code)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                            pin_memory=False, drop_last=False)
    y_val, y_pred = validate(model, val_loader)
    return y_val, y_pred

def eval(y_val, y_pred):
    #val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    val_df["pred"] = val_df["pct_rank"]
    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
    y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
    return score

num_models = args.num_models

if num_models == 2:
    y_val, y_pred_1 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_1, args.model_ckp_1_vbl_code)
    print("model 1 pred", eval(y_val, y_pred_1))
    y_val, y_pred_2 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_2, args.model_ckp_2_vbl_code)
    print("model 2 pred", eval(y_val, y_pred_2))
    y_pred = (y_pred_1 + y_pred_2)/num_models
    print("avg model pred", eval(y_val, y_pred))
elif num_models == 3:   
    y_val, y_pred_1 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_1, args.model_ckp_1_vbl_code)
    print("model 1 pred", eval(y_val, y_pred_1))
    y_val, y_pred_2 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_2, args.model_ckp_2_vbl_code)
    print("model 2 pred", eval(y_val, y_pred_2))
    y_val, y_pred_3 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_3, args.model_ckp_3_vbl_code)
    print("model 3 pred", eval(y_val, y_pred_3))
    y_pred = (y_pred_1 + y_pred_2 + y_pred_3)/num_models
    print("avg model pred", eval(y_val, y_pred))
elif num_models == 4:    
    y_val, y_pred_1 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_1, args.model_ckp_1_vbl_code)
    print("model 1 pred", eval(y_val, y_pred_1))
    y_val, y_pred_2 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_2, args.model_ckp_2_vbl_code)
    print("model 2 pred", eval(y_val, y_pred_2))
    y_val, y_pred_3 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_3, args.model_ckp_3_vbl_code)
    print("model 3 pred", eval(y_val, y_pred_3))
    y_val, y_pred_4 = predict(args.model_name_or_path, args.model_ckp_path, args.model_ckp_4, args.model_ckp_4_vbl_code)
    print("model 4 pred", eval(y_val, y_pred_4))
    y_pred = (y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4)/num_models
    print("avg model pred", eval(y_val, y_pred))

# define weights to consider
w = np.linspace(1.0, 0.0, num=args.steps)
best_score, best_weights = 0.0, None
total_len = int(sum([len(i) for i in product(w, repeat=num_models)])/num_models)
# iterate all possible combinations (cartesian product)
for weights in tqdm(product(w, repeat=num_models), total=total_len):
    if np.all(np.array(weights) == np.array(weights[0])): # skip if all values are the same
        continue
    if not np.any(np.array(weights) == 1.0): # skip if 1 is not present
        continue
    if num_models == 2:
        y_pred = ((y_pred_1*weights[0]) + (y_pred_2*weights[1]))/np.sum(weights)
    elif num_models == 3:
        y_pred = ((y_pred_1*weights[0]) + (y_pred_2*weights[1]) + (y_pred_3*weights[2]))/np.sum(weights)
    elif num_models == 4:    
        y_pred = ((y_pred_1*weights[0]) + (y_pred_2*weights[1]) + (y_pred_3*weights[2]) + (y_pred_4*weights[3]))/np.sum(weights)
    score = eval(y_val, y_pred)
    if score > best_score:
        best_score, best_weights = score, weights
        print('>%s %.6f' % (best_weights, best_score))
