import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import json
def plot_bar(df, name, save_name = None, need_to_save = True):
    if save_name == None:
        save_name = name
    p = df[name].value_counts()
    x = p.index.to_numpy()
    y = p.to_numpy()
    plt.figure()
    plt.title(save_name.split('/')[-1], fontproperties="SimHei")
    plt.bar(x, y)
    if need_to_save == True:
        plt.savefig("./Feature Figures/" + save_name + ".png")
    plt.show()

def target_encoding(name, df, m=1):
    df[name] = df[name].str.split(";")
    df = df.explode(name)
    overall = df["label"].mean()
    df = df.groupby(name).agg(
        freq=("label", "count"),
        in_category=("label", np.mean)
    ).reset_index()
    df["weight"] = df["freq"] / (df["freq"] + m)
    df["score"] = df["weight"] * df["in_category"] + (1 - df["weight"]) * overall
    return df

def get_id_score(id_list, id_score):
    x = sorted(id_list.split(";"))
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x) if k != "nan"}
    if x_count:
        x_sort = sorted(x_count.items(), key=lambda k: -k[1])
        top_x = x_sort[:3]
        res = [(n, id_score.get(k, 0)) for k, n in top_x]
        res = sum(n*v for n, v in res) / sum(n for n, v in res)
        return res
    else:
        return np.nan

def load_seq_feature(df, columns):
    for name in columns:
        df[name] = df[name].apply(lambda x: json.loads(x) if not pd.isna(x) else x)
    return df
