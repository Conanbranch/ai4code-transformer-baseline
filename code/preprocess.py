import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='process arguments')

parser.add_argument('--num_sample', type=int, default=20, help='number of code cells to sample')
parser.add_argument('--sample_data', type=float, default=1.0, help='proportion the data for training and validation set')

args = parser.parse_args()

data_dir = Path('..//input/')
if not os.path.exists("./data"):
    os.mkdir("./data")

def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=path.stem)
            .rename_axis('cell_id')
    )


paths_train = list((data_dir / 'train').glob('*.json'))
notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
]
df = (
    pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
)

df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()  # Split the string representation of cell_ids into a list


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}
df_ranks = (
    pd.DataFrame
        .from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
)

# original
df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
df["pct_rank_old"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")
#df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

# new ranking
df = df.sort_values(['id','rank'],ascending=True).reset_index(drop=True)
df["ct_rank"] = df.groupby(["id", "cell_type"]).cumcount() + 1 # this will shift the first value to 0.5 and duplicate the orignal pred
df["mod_rank"] = df.loc[df['cell_type'] == 'code']['ct_rank']
df["mod_rank"] = df.groupby(["id"])["mod_rank"].fillna(method='ffill')
df["dup_rank"] = df.loc[df['cell_type'] == 'markdown'].groupby(["id", "cell_type", "mod_rank"]).cumcount() + 1
df["dup_count"] = df.loc[df['cell_type'] == 'markdown'].groupby(["id", "cell_type", "mod_rank"])["mod_rank"].transform("count")
df["t_mod_rank"] = df.loc[(df['cell_type'] == 'markdown')]["mod_rank"] + (df.loc[(df['cell_type'] == 'markdown')]["dup_rank"] / (df.loc[(df['cell_type'] == 'markdown')]["dup_count"] + 1))
df.t_mod_rank.fillna(df.mod_rank, inplace=True)
df["mod_rank_1"] = df.groupby(["id"])["t_mod_rank"].fillna(method='bfill')
df["dup_rank_1"] = df.loc[(df['cell_type'] == 'markdown') & (df.mod_rank_1 == 1.0)].groupby(["id", "cell_type", "mod_rank_1"]).cumcount() + 1
df["dup_count_1"] = df.loc[(df['cell_type'] == 'markdown') & (df.mod_rank_1 == 1.0)].groupby(["id", "cell_type", "mod_rank_1"])["mod_rank_1"].transform("count")
df["mod_rank_2"] = df.loc[(df['cell_type'] == 'markdown') & (df.mod_rank_1 == 1.0)]["mod_rank_1"] - ((df.loc[(df['cell_type'] == 'markdown') & (df.mod_rank_1 == 1.0)]["dup_count_1"] + 1) - (df.loc[(df['cell_type'] == 'markdown') & (df.mod_rank_1 == 1.0)]["dup_rank_1"])) / (df.loc[(df['cell_type'] == 'markdown') & (df.mod_rank_1 == 1.0)]["dup_count_1"] + 1)
df.t_mod_rank.fillna(df.mod_rank_2, inplace=True)
df["mod_rank"] = df["t_mod_rank"]
df["count"] = df.loc[df['cell_type'] == 'code'].groupby(["id", "cell_type"])["mod_rank"].transform("count") # + 1
df["count"] = df.groupby(["id"])["count"].fillna(method='bfill').fillna(method='ffill')
df["pct_rank"] = df["mod_rank"] / df["count"]
df = df.drop(columns = ["count","dup_rank","dup_rank_1","t_mod_rank","mod_rank_1","dup_count","dup_count_1","mod_rank_2"])

from sklearn.model_selection import GroupShuffleSplit

NTRAIN = 0.9 * args.sample_data # proportion of training set
NVALID = 0.1 * args.sample_data # proportion of validation set

splitter = GroupShuffleSplit(n_splits=1, train_size=NTRAIN, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)

# Base markdown dataframes
train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)
train_df_mark.to_csv("./data/train_mark.csv", index=False)
val_df_mark.to_csv("./data/val_mark.csv", index=False)
val_df.to_csv("./data/val.csv", index=False)
train_df.to_csv("./data/train.csv", index=False)


# Additional code cells
def clean_code(cell):
    return str(cell).replace("\\n", "\n")


def sample_cells(cells, n):
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def get_features(df):
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, args.num_sample)
        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
        features[idx]["codes"] = codes
    return features

val_fts = get_features(val_df)
json.dump(val_fts, open("./data/val_fts.json","wt"))
train_fts = get_features(train_df)
json.dump(train_fts, open("./data/train_fts.json","wt"))
