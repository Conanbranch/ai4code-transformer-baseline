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
parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv', help='path for markdown training data')
parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json', help='path for code training data')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv', help='path for markdown validation data')
parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json', help='path for code validation data')
parser.add_argument('--val_path', type=str, default="./data/val.csv", help='path for validation data')
parser.add_argument('--model_ckp_path', type=str, default="./output", help='path for model and model checkpoints')
parser.add_argument('--model_ckp', type=str, default="model.pt", help='model checkpoint filename')
parser.add_argument('--model', type=str, default="model.bin", help='model filename')

parser.add_argument('--md_max_len', type=int, default=64, help='maximum length of tokenized markdown')
parser.add_argument('--total_max_len', type=int, default=512, help='maximum length of tokenized markdown and code')
parser.add_argument('--batch_size', type=int, default=8, help='training batchsize, try --batch_size 8 if you encounter memory issues')
parser.add_argument('--accumulation_steps', type=int, default=4, help='number of accumulating steps')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs, 3 or 5 are good starting points')
parser.add_argument('--n_workers', type=int, default=8, help='number of workers')
parser.add_argument('--re_init', type=bool, default=False, help="option to re-initialize layers of the pretrained model")
parser.add_argument('--reinit_n_layers', type=int, default=0, help="number of layers of the pretrained model to re-initialize")
parser.add_argument('--resume_train', type=bool, default=False, help="resume training if previous training was interupted")
parser.add_argument('--correct_bias', type=bool, default=False, help="include bias correction")
parser.add_argument('--code_sep_token', type=bool, default=True, help="include seperator tokens between code samples")
parser.add_argument('--pad_between_code', type=bool, default=True, help="include seperator tokens between code samples")
parser.add_argument('--vbl_code', type=bool, default=False, help="use variable length code")
parser.add_argument('--lr', type=float, default=3e-5, help="learning rate")
parser.add_argument('--wd', type=float, default=0.01, help="weight_decay")
parser.add_argument('--wup', type=float, default=0.05, help="warm up rate")

args = parser.parse_args()
    
if not os.path.exists("./output"):
    os.mkdir("./output")      
      
data_dir = Path('..//input/')

train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(args.train_features_path))
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

train_ds = MarkdownDataset(train_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts, code_sep_token = args.code_sep_token, 
                           pad_between_code = args.pad_between_code, vbl_code=args.vbl_code)
val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts, code_sep_token = args.code_sep_token, 
                         pad_between_code = args.pad_between_code, vbl_code=args.vbl_code)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)

def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir + '/' + args.model_ckp
    #f_path = checkpoint_dir + '/model_new_rank_01_v2.pt'
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath + '/' + args.model_ckp)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']

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

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    
    # optimizer and lr schedulers, includes weight decay
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      correct_bias=args.correct_bias)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.wup * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
    
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    
    epoch = 0
    
    if args.resume_train == True:
        model, optimizer, scheduler, epoch = load_ckp(args.model_ckp_path, model, optimizer, scheduler)
    
    for e in range(epoch,epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
        
        torch.save(model.state_dict(), args.model_ckp_path + "/" + "epoch_" + str(e + 1) + "_" + args.model)

        checkpoint = {
          'epoch': e + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict()
        }
        
        save_ckp(checkpoint, args.model_ckp_path)

        y_val, y_pred = validate(model, val_loader)
        print("val loss (markdown)",  np.round(mean_absolute_error(y_val, y_pred),4))
        #val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df["pred"] = val_df["pct_rank"]
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        print("pred score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        y_dummy = val_df.loc[val_df["cell_type"] == "markdown"].sort_values("pred").groupby('id')['cell_id'].apply(list)
        print("md pred score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        y_dummy = val_df.loc[val_df["cell_type"] == "code"].sort_values("pred").groupby('id')['cell_id'].apply(list)
        print("code pred score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

    torch.save(model.state_dict(), args.model_ckp_path + "/" + args.model)
    
    return model, y_pred

model = MarkdownModel(args.model_name_or_path, args.re_init, args.reinit_n_layers)
model = model.cuda()
model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)
