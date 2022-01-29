import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

def draw_save(x, y, save_path, x_label = "global_step", y_label = "val_loss"):
    plt.figure(1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(y_label)
    plt.plot(x, y)
    plt.savefig(save_path + "/" + y_label + ".png")
    plt.clf()

def post_proprecess(path, save_dir):
    df = pd.read_csv(path, header = None)
    df.columns = ["user_id", "pred"]
    df["pred"] = df["pred"].apply(lambda x: max(0.0, min(7.0, x)))
    df["pred"] = df["pred"].apply(score_cut) # 截断处理
    inactive = pd.read_csv("./preprocessed_data/inactive_user.csv")["user_id"].to_list()
    inactive_index = df[df.user_id.isin(inactive)].index
    df.loc[inactive_index, "pred"] = 0
    df.to_csv(save_dir + "baseline_submission_post_preprocess.csv", index = False, header = False, float_format="%.2f")


def load_seq_feature(df, columns):
    for name in columns:
        df[name] = df[name].apply(lambda x: json.loads(x) if not pd.isna(x) else x)
    return df

def score_cut(x):
    y = int(x)
    if x > y + 0.5:
        return float(min(7, y + 1))
    elif x == y + 0.5:
        return x
    else:
        return float(max(0, y))


def post_proprecess2(path, save_dir):
    df = pd.read_csv(path, header=None)
    df.columns = ["user_id", "pred"]
    df["pred"] = df["pred"].apply(lambda x: max(0.0, min(7.0, x)))
    df["pred"] = df["pred"].apply(score_cut)
    inactive = pd.read_csv("./preprocessed_data/inactive_user.csv")["user_id"].to_list()
    inactive_index = df[df.user_id.isin(inactive)].index
    df.loc[inactive_index, "pred"] = 0
    df.to_csv(save_dir + "baseline_submission_post_preprocess2.csv", index=False, header=False, float_format="%.2f")
