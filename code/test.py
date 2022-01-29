'''
随机种子1998，效果最好，b榜86.3012
'''

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import gc
import pandas as pd
import numpy as np
import json
import math
import paddle
from paddle.io import DataLoader, Dataset
import time, shutil
# from util import draw_save, post_proprecess, load_seq_feature
from tqdm import tqdm
import warnings
from paddle_models import Model_Sliding_Window, Dataset_Sliding_Window
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
#############
learning_rate = 0.001
total_epoch = 15
train_batch_size = 128
val_batch_size = 128
val_data_num = 24000


#############

SEED = 1998


def post_proprecess(path, save_dir):
    df = pd.read_csv(path, header=None)
    df.columns = ["user_id", "pred"]
    df["pred"] = df["pred"].apply(lambda x: max(0.0, min(7.0, x)))
    df["pred"] = df["pred"].apply(score_cut)  # 截断处理
    inactive = pd.read_csv("./preprocessed_data/inactive_user.csv")["user_id"].to_list()
    inactive_index = df[df.user_id.isin(inactive)].index
    df.loc[inactive_index, "pred"] = 0
    df.to_csv(save_dir + "baseline_submission_post_preprocess_{}.csv".format(SEED), index=False, header=False, float_format="%.2f")


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


@paddle.no_grad()
def predict(model, test_loader, test_data, weight_path, save_dir):
    model.set_dict(paddle.load(weight_path)["model"])  # AUC指标
    model.eval()
    test_pred = []
    for launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat, _ in tqdm(test_loader):
        pred = model(launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat)
        test_pred.extend(pred.detach().cpu().numpy().sum(axis=1).flatten())

    test_data["prediction"] = test_pred
    test_data = test_data[["user_id", "prediction"]]

    # test_data.to_csv(save_dir + "baseline_submission.csv", index=False, header=False, float_format="%.2f")
    test_data.to_csv('./output/only-test/' + "baseline_submission_{}.csv".format(SEED), index=False, header=False,
                     float_format="%.2f")  # 保存时把随机种子也加到名字里

    post_proprecess('./output/only-test/' + "baseline_submission_{}.csv".format(SEED), save_dir)


if __name__ == "__main__":
    seed = 20211225
    paddle.seed(seed)
    np.random.seed(seed)

    data_dir = "./preprocessed_data/Dynamic-sliding-window/v1/"

    seq_feature = ["launch_seq", "launch_type_seq", "playtime_seq", "video_num_seq",
                   "duration_prefer", "interact_prefer"]

    test_data = pd.read_csv(data_dir + "test_data.txt", sep="\t")
    test_data = load_seq_feature(test_data, seq_feature)

    test_dataset = Dataset_Sliding_Window(test_data)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    model = Model_Sliding_Window()

    predict(model, test_loader, test_data, './weights/BL_paddle.pdparams', './output/only-test/')  # 86.3012
