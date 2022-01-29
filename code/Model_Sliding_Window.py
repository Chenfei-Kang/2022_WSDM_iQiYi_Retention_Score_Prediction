import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import gc
import pandas as pd
import numpy as np
import json
import math
import paddle
from paddle.io import DataLoader, Dataset
import time, shutil
from util import draw_save, post_proprecess, load_seq_feature
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

save_name = "Model_Sliding_Window"
save_file = "./feature_engineering_sliding_window_1.py"
code_file = ["./Model_Sliding_Window.py",  "./feature_engineering_sliding_window_1.py", "./paddle_models.py"]
#############

@paddle.no_grad()
def validate(model, val_loader, optimizer, criterion):
    model.eval()
    val_loss = []
    val_pres = []
    true_labels = []
    for launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat, label_seq in val_loader:
        pred = model(launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat)

        loss = criterion(pred, label_seq)
        val_pres.extend(pred.detach().cpu().numpy())
        true_labels.extend(label_seq.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        val_loss.append(loss.item())
    val_pres = np.vstack(val_pres)
    true_labels = np.vstack(true_labels)
    val_score = 1 - np.average(np.abs(val_pres.sum(axis = 1).flatten() - true_labels.sum(axis = 1).flatten())/7)

    val_auc = roc_auc_score(true_labels.tolist(), val_pres.tolist())

    model.train()
    return np.mean(val_loss), val_score, val_auc
@paddle.no_grad()
def predict(model, test_loader, test_data, save_dir):
    model.set_dict(paddle.load(save_dir + "./BL_paddle.pdparams")["model"]) # AUC指标
    model.eval()
    test_pred = []
    for launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat, _ in tqdm(test_loader):
        pred = model(launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat)
        test_pred.extend(pred.detach().cpu().numpy().sum(axis=1).flatten())

    test_data["prediction"] = test_pred
    test_data = test_data[["user_id", "prediction"]]

    test_data.to_csv(save_dir + "baseline_submission.csv", index=False, header=False, float_format="%.2f")
    test_data.to_csv('./output/' + "baseline_submission.csv", index=False, header=False, float_format="%.2f")
    
    post_proprecess(save_dir + "baseline_submission.csv", save_dir)


def train(model, optimizer, criterion, train_loader, val_loader):
    train_loss_log = []
    val_loss_log = []
    val_score_log = []
    val_auc_log = []
    eval_step = int(len(train_loader) / 4)
    global_step = 0

    best_results = {"best_val_loss": 100, "best_val_score": 0, "best_val_auc": 0}

    best_state = {}
    information_log = []
    # val_loss, val_score, val_auc = -1, -1, -1
    for epoch in range(total_epoch):
        # train_loss = train(model, train_loader, optimizer, criterion)
        # val_loss = validate(model, val_loader, optimizer, criterion)
        model.train()
        train_results = {"train_loss": [], "train_score": [], "train_auc": []}
        val_results = {"val_loss": 100, "val_score": 0, "val_auc": 0}
        i = 0

        for launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat, label_seq in tqdm(train_loader):
            pred = model(launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat)

            # pred = pred.squeeze(1)
            loss = criterion(pred, label_seq)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            # if i % 100 == 0:
            #     print(i, loss.detach().cpu().numpy().item())

            train_results["train_loss"].append(loss.detach().cpu().numpy().item())

            # 需要统一记录一下Pred向量进行总体train的计算
            # true_labels = label_seq.detach().cpu().numpy()
            # train_pred = pred.detach().cpu().numpy()
            # train_results["train_score"].append(1 - np.average(np.abs( train_pred- true_labels) / 7))
            #
            # train_auc = roc_auc_score(true_labels.tolist(), train_pred.tolist())
            # train_results["train_auc"].append(train_auc)

            if global_step % eval_step == 0 and global_step >= 0:
                val_loss, val_score, val_auc = validate(model, val_loader, optimizer, criterion)
                val_results = {"val_loss": val_loss, "val_score": val_score, "val_auc": val_auc}
                val_loss_log.append([global_step, val_loss])
                val_score_log.append([global_step, val_score])
                val_auc_log.append([global_step, val_auc])
                # print("step {} of epoch {}, val loss:{:.6f}, val score:{:.6f} \n"
                #       .format(global_step, epoch, val_loss, val_score))
                information_log.append("step {} of epoch {}, val loss:{:.6f}, val score:{:.6f} \n"
                                       .format(global_step, epoch, val_loss, val_score))
                # if val_score > best_val_score:
                if val_auc > best_results["best_val_auc"]:
                    best_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                  'epoch': epoch, 'lr': learning_rate,
                                  'batch_size': train_batch_size, "global_step": global_step}
                    best_results = {"best_val_loss": val_loss, "best_val_score": val_score, "best_val_auc": val_auc}

                    information_log.append(
                        "best occur at step {} of epoch {}, val loss:{:.6f}, val score:{:.6f}, val_auc:{:.6f} \n"
                        .format(global_step, epoch, val_loss, val_score, val_auc))
            i += 1
            global_step += 1
            # gc.collect()
        ave_train_loss = np.mean(train_results["train_loss"])

        train_loss_log.append([global_step, ave_train_loss])
        # if epoch == 10:
        #     optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate / 10)
        print(
            "epoch:{}, train loss:{:.6f}, val loss:{:.6f}, val score:{:.6f}, val auc:{:.6f}, best loss:{:.6f}, best score:{:.6f}, best auc:{:.6f}".format(
                epoch, ave_train_loss, val_results["val_loss"], val_results["val_score"], val_results["val_auc"],
                best_results["best_val_loss"], best_results["best_val_score"], best_results["best_val_auc"]))
        information_log.append(
            "epoch:{}, train loss:{:.6f}, val loss:{:.6f}, val score:{:.6f}, val auc:{:.6f}, best loss:{:.6f}, best score:{:.6f}, best auc:{:.6f}".format(
                epoch, ave_train_loss, val_results["val_loss"], val_results["val_score"], val_results["val_auc"],
                best_results["best_val_loss"], best_results["best_val_score"], best_results["best_val_auc"]))

        # 存储训练日志和结果
    save_dir = "./checkpoints/paddle/" + save_name + " " + time.strftime("%m-%d %H-%M-%S", time.localtime())
    os.makedirs(save_dir)
    save_dir += "/"
    for file in code_file:
        shutil.copy(file, save_dir + file.split('/')[-1])
    paddle.save(best_state, save_dir + "BL_paddle.pdparams")
    f = open(save_dir + "log.txt", mode='w')
    for info in information_log:
        f.write(info)
    f.close()
    train_loss_log = np.array(train_loss_log)
    val_loss_log = np.array(val_loss_log)
    val_score_log = np.array(val_score_log)
    val_auc_log = np.array(val_auc_log)

    draw_save(val_loss_log[:, 0], val_loss_log[:, 1], save_path=save_dir, y_label="val_loss")
    draw_save(train_loss_log[:, 0], train_loss_log[:, 1], save_path=save_dir, y_label="train_loss")
    draw_save(val_score_log[:, 0], val_score_log[:, 1], save_path=save_dir, y_label="val_score")
    draw_save(val_auc_log[:, 0], val_auc_log[:, 1], save_path=save_dir, y_label="val_auc")
    return save_dir

if __name__ == "__main__":
    seed = 20211225
    paddle.seed(seed)
    np.random.seed(seed)

    data_dir = "./preprocessed_data/Dynamic-sliding-window/v0/"
    test_data_dir = "./preprocessed_data/Dynamic-sliding-window/v1/"
    # data_dir = '../tang/temp_data/'
    # train data
    data = pd.read_csv(data_dir + "train_data.txt", sep="\t")
    seq_feature = ["launch_seq", "launch_type_seq", "playtime_seq","video_num_seq",
                   "duration_prefer", "interact_prefer"]
    data = load_seq_feature(data, seq_feature + ["label_seq"])
    # shuffle data
    data = data.sample(frac = 1).reset_index(drop=True)

    train_dataset = Dataset_Sliding_Window(data.iloc[:-val_data_num])
    val_dataset = Dataset_Sliding_Window(data.iloc[-val_data_num:])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=12)

    test_data = pd.read_csv(test_data_dir + "test_data.txt", sep="\t")
    test_data = load_seq_feature(test_data, seq_feature)

    test_dataset = Dataset_Sliding_Window(test_data)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    model = Model_Sliding_Window()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate)
    criterion = paddle.nn.MSELoss()

    save_dir = train(model, optimizer, criterion, train_loader, val_loader)
    # save_dir = './checkpoints/paddle/Model_Sliding_Window 01-06 04-51-18/'
    predict(model, test_loader, test_data, save_dir)






