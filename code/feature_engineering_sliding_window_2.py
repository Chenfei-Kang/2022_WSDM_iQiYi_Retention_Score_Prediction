import pandas as pd
import numpy as np
from tools import plot_bar, target_encoding, get_id_score, load_seq_feature
from itertools import groupby
import json



#%% part1 label split
random_seed = 1998      # 20211225
np.random.seed(random_seed)

# 选取end_date、产生label、产生过去selected_span天数的活跃序列

input_dir = "./data/"
output_dir = "./preprocessed_data/Dynamic-sliding-window/v1/"    # 这里的v0和v1暂时只是随机种子不同
launch = pd.read_csv(input_dir + "app_launch_logs.csv")
print(launch["date"].min(), launch["date"].max())
launch_gb = launch.groupby(by = "user_id").agg(launch_date=("date", list),
    launch_type=("launch_type", list)).reset_index()



launch_gb["max_date"] = launch_gb.apply(lambda x: np.max(x["launch_date"]), axis = 1)
launch_gb["min_date"] = launch_gb.apply(lambda x: np.min(x["launch_date"]), axis = 1)
launch_gb["span"] = launch_gb.apply(lambda x: max(x["max_date"], 161) - min(x["min_date"],131), axis = 1)
launch_gb["launch_sum"] = launch_gb.apply(lambda x: len(x["launch_date"]), axis = 1)
# plot_bar(launch_gb, "max_date", "最大日期分布")
# plot_bar(launch_gb, "min_date", "最小日期分布")
# plot_bar(launch_gb, "span", "活跃日期跨度分布")

# old_data = pd.read_csv("../ref_code/A_Shui/wsdm_model_data/train_data.txt", delimiter='\t')
# plot_bar(old_data, "end_date", "基线的end_date分布")
# plot_bar(old_data, "label", "基线的label分数分布")


# 训练数据选取、end_date构造
inactive_user = launch_gb[launch_gb["launch_sum"] <= 1].loc[:, ["user_id"]]
launch_gb_active = launch_gb.drop(inactive_user.index).reset_index(drop=True).loc[:,
                   ["user_id", "launch_date", "launch_type"]]
illegal_num, less_num = 0, 0

def choose_end_date_sliding_window(launch_date):
    global  illegal_num
    n1, n2 = min(launch_date), max(launch_date)
    # if n1 < n2 - 7:
    #     end_date = np.random.randint(n1, n2 - 7)
    # else:
    #     end_date = np.random.randint(100, 222 - 7)
    # return end_date
    end_date = -1
    if n2 - 6 >= 161:
        end_date = np.random.randint(max(n2 - 18, 160), n2 - 6)
    else:
        # 选取的end_date不能大于160，这些数据可能对测试集的意义不大(测试集的enda_date均大于160)
        if n2 > 160:
            end_date = np.random.randint(n2 - 14, n2 - 6)
        else:
            # end_date = np.random.randint(160 - 18, 160 - 6)
            end_date = np.random.randint(161, 222)
    return end_date

def choose_start_date_sliding_window(row):
    n1, end_date = min(131, min(row["launch_date"])), row["end_date"]
    # if end_date - 131 + 1 > 60:
    #     return end_date - 59
    # else:
    #     return 131
    if end_date - n1 + 1 > 60:
        return end_date - 59
    else:
        return n1


launch_gb_active["end_date"] = launch_gb_active.launch_date.apply(choose_end_date_sliding_window)
launch_gb_active["start_date"] = launch_gb_active.apply(choose_start_date_sliding_window, axis = 1)
print("不合法的数目:", illegal_num)
# launch_gb_active = launch_gb_active[launch_gb_active["end_date"] != -1]

def get_label(row):
    launch_list = set(row.launch_date)
    end = row.end_date
    label = []
    for x in range(end + 1, end + 8):
        if x in launch_list:
            label.append(1)
        else:
            label.append(0)
    return label
launch_gb_active["label_seq"] = launch_gb_active.apply(get_label, axis=1) # 每个Label是长为7的向量
launch_gb_active["label"] = launch_gb_active["label_seq"].apply(lambda x: np.sum(x))
launch_gb_active["selected_span"] = launch_gb_active.apply(lambda x: x.end_date - x.start_date, axis = 1)
plot_bar(launch_gb_active, "end_date", "sliding_window/v2_采样后的end_date分布")
plot_bar(launch_gb_active, "start_date", "sliding_window/v2_采样后的start_date分布")
plot_bar(launch_gb_active, "selected_span", "sliding_window/v2_采样后的selected_span分布")
plot_bar(launch_gb_active, "label", "sliding_window/v2_采样后的label分数分布")

# launch_gb_active.loc[:, ["user_id", "end_date", "label", "label_score"]]\
#     .to_csv("./data/launch_gb_active.txt", sep = '\t', index = False)


# test_data start_date选取
test = pd.read_csv("./data/" + "test-B.csv")
test_with_launch_seq = test.merge(launch_gb[["user_id", "launch_date"]], how = "left", on = "user_id")
test_with_launch_seq["start_date"] = test_with_launch_seq.apply(choose_start_date_sliding_window, axis = 1)
plot_bar(test_with_launch_seq, "start_date", "sliding_window/v2_验证集的start_date分布")
test_with_launch_seq["label"] = -1
test_with_launch_seq["label_seq"] = -1


# 合并train和test
train = launch_gb_active[["user_id", "start_date", "end_date", "label", "label_seq"]]
test  = test_with_launch_seq[["user_id", "start_date", "end_date", "label", "label_seq"]]
total_data = pd.concat([train, test], ignore_index=True)


# test中的user_id在train中全部出现，但是有43个用户不活跃(登录次数 <= 1)
# 先获取一下登录的历史信息,再添加到total_data的列属性中，方便之后产生活跃序列
launch_history_info = launch_gb[["user_id", "launch_date", "launch_type"]]
previous_shape = total_data.shape[0]
total_data = total_data.merge(launch_history_info, how = "left", on = "user_id")
print("total_data merge launch_history_info ", previous_shape == total_data.shape[0])

# 产生过去selected_span天的活跃序列
# test中的user_id在train中全部出现，但是有43个用户不活跃(登录次数 <= 1)
# selected_span = 32
# 0 for not launch, 1 for type 0, 2 for type 1
def get_launch_type_seq(row):
    if not isinstance(row.launch_type, list) and pd.isna(row.launch_type): # 不活跃的用户处理为nan
        return np.nan
    seq_sort = sorted(zip(row.launch_type, row.launch_date), key = lambda x: x[1])
    seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])} # 同一天登录类型的取最大的
    start, end = row.start_date, row.end_date
    seq = [seq_map.get(x, 0) for x in range(start, end + 1)] # 把活跃的序列替换为登录的类型序列
    return seq # 先不padding
def get_launch_seq(row):
    if not isinstance(row.launch_date, list) and pd.isna(row.launch_date):
        return np.nan
    launch_list = row.launch_date
    seq = [0]*(row.end_date - row.start_date + 1)
    start, end = row.start_date, row.end_date
    for i in range(start, end + 1):
        if i in launch_list:
            seq[i - start] = 1
    return seq

total_data["launch_type_seq"] = total_data.apply(get_launch_type_seq, axis=1)
total_data["launch_seq"] = total_data.apply(get_launch_seq, axis = 1)
total_data_with_launch_seq = total_data

print(total_data_with_launch_seq.shape[0] == total_data.shape[0])

total_data_with_launch_seq[['user_id', "start_date", 'end_date', 'label', 'label_seq', "launch_seq", 'launch_type_seq']].\
    to_csv(output_dir + "total_data_with_launch_seq.txt", sep = "\t", index = False)
# total_data_with_launch_seq = total_data_with_launch_seq[['user_id', "start_date", 'end_date', 'label',
#                                                          'label_seq', "launch_seq", 'launch_type_seq']]

#%% part2 behavior process
# input_dir = "../wsdm_train_data/"
# output_dir = "../preprocessed_data/v0/"
# 产生四个活跃序列
# launch_seq 0-没有登录  1-登录
# launch_type_seq 0-没有登录 1和2代表已经登录
# playtime_seq 每天的播放时长
# video_num_seq 每天观看的视频总数
# interaction_num_seq 每天互动的总数

#launch_history_seq 在label_split.py里已经产生

# total_data_with_launch_seq = pd.read_csv(output_dir + "total_data_with_launch_seq.txt", sep = '\t')
# total_data_with_launch_seq["launch_seq"] = total_data_with_launch_seq.launch_seq.apply(lambda x: json.loads(x))

print(total_data_with_launch_seq['launch_type_seq'])

# play_time_seq 每天的播放时长
playback = pd.read_csv(input_dir + "user_playback_data.csv", dtype={"item_id": str})
playback = playback.merge(total_data_with_launch_seq[["user_id", "start_date", "end_date", "label"]], how="inner", on="user_id")
playback = playback.loc[(playback.date >= playback.start_date) & (playback.date <= playback.end_date)]

video_data = pd.read_csv(input_dir + "video_related_data.csv", dtype=str)
previous_shape = playback.shape[0]
playback = playback.merge(video_data[video_data.item_id.notna()], how="left", on="item_id")
print(playback.shape[0] == previous_shape)

playback_gb = playback.groupby(["user_id", "start_date", "end_date", "label"]).agg(
    item_id_list = ("item_id", list),
    playtime_list=("playtime", list),
    date_list=("date", list),
    duration_list=("duration", lambda x: ";".join(map(str, x))),
    father_id_list=("father_id", lambda x: ";".join(map(str, x))),
    tag_list=("tag_list", lambda x: ";".join(map(str, x))),
    cast_list=("cast", lambda x: ";".join(map(str, x)))
).reset_index()

def get_playtime_seq(row):
    seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
    seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
    seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
    seq = [round(seq_norm.get(i, 0), 4) for i in range(row.start_date, row.end_date + 1)]
    return seq
playback_gb["playtime_seq"] = playback_gb.apply(get_playtime_seq, axis=1)

# video_num_seq 每天观看的视频总数序列
def get_video_num_seq(row):
    scale = 50 # 大部分取值都在0~50
    seq_sort = sorted(zip(row.item_id_list, row.date_list), key = lambda x: x[1])
    seq_map = {}
    for k, g in groupby(seq_sort, key=lambda x: x[1]):
        num = 0
        for i in g:
            num += 1
        seq_map[k] = num/scale
    seq = [seq_map.get(i, 0) for i in range(row.start_date, row.end_date + 1)]
    return seq
playback_gb["video_num_seq"] = playback_gb.apply(get_video_num_seq, axis = 1)



previous_shape = total_data_with_launch_seq.shape[0]
total_data_with_all_seq = total_data_with_launch_seq.merge(
    playback_gb[["user_id", "start_date", "end_date", "label", "playtime_seq", "video_num_seq"]], how = "left",
    on = ["user_id", "start_date", "end_date", "label"])

total_data_with_all_seq.to_csv(output_dir + "total_data_with_all_seq.txt", sep = '\t', index = False)


# num = {}
# def get_item_num_distribution(df):
#     global num
#     for i in range(df.shape[0]):
#         if df.loc[i, "label"] != -1:
#             if isinstance(df.loc[i, "video_num_seq"], list):
#                 for j in df.loc[i, "video_num_seq"]:
#                     if j != 0:
#                         if num.get(j) is None:
#                             num[j] = 1
#                         else:
#                             num[j] += 1
# get_item_num_distribution(total_data_with_launch_seq)
#%% part3 static feature process

# 视频的静态信息: father_id_score、tag_id_score、cast_id_score和时长偏好duration_prefer
playback = pd.read_csv(input_dir + "user_playback_data.csv", dtype={"item_id": str})
playback = playback.merge(total_data_with_all_seq[["user_id", "start_date", "end_date", "label"]], how="inner", on="user_id")
playback = playback.loc[(playback.date >= playback.start_date) & (playback.date <= playback.end_date)]

video_data = pd.read_csv(input_dir + "video_related_data.csv", dtype=str)
previous_shape = playback.shape[0]
playback = playback.merge(video_data[video_data.item_id.notna()], how="left", on="item_id")


# 对所有的视频信息: father_id、tag_id 和 cast_id 进行 target_encoding
df = playback.loc[(playback.label >= 0) & (playback.father_id.notna()), ["father_id", "label"]]
father_id_score = target_encoding("father_id", df)

df = playback.loc[(playback.label >= 0) & (playback.tag_list.notna()), ["tag_list", "label"]]
tag_id_score = target_encoding("tag_list", df)
tag_id_score.rename({"tag_list": "tag_id"}, axis=1, inplace=True)

df = playback.loc[(playback.label >= 0) & (playback.cast.notna()), ["cast", "label"]]
cast_id_score = target_encoding("cast", df)
cast_id_score.rename({"cast": "cast_id"}, axis=1, inplace=True)


playback_gb = playback.groupby(["user_id", "start_date", "end_date", "label"]).agg(
    playtime_list=("playtime", list),
    date_list=("date", list),
    duration_list=("duration", lambda x: ";".join(map(str, x))),
    father_id_list=("father_id", lambda x: ";".join(map(str, x))),
    tag_list=("tag_list", lambda x: ";".join(map(str, x))),
    cast_list=("cast", lambda x: ";".join(map(str, x)))
).reset_index()

# 对duration 进行16维度向量的编码
drn_desc = video_data.loc[video_data.duration.notna(), "duration"].astype(int)
def get_duration_prefer(duration_list):
    drn_list = sorted(duration_list.split(";"))
    drn_map = {k: sum(1 for _ in g) for k, g in groupby(drn_list) if k != "nan"}
    if drn_map:
        max_ = max(drn_map.values())
        res = [round(drn_map.get(str(i), 0)/max_, 4) for i in range(1, 17)]
        return res
    else:
        return np.nan
playback_gb["duration_prefer"] = playback_gb.duration_list.apply(get_duration_prefer)

id_score = dict()
id_score.update({x[1]: x[5] for x in father_id_score.itertuples()})
id_score.update({x[1]: x[5] for x in tag_id_score.itertuples()})
id_score.update({x[1]: x[5] for x in cast_id_score.itertuples()})

# check if features ids are duplicated
print(father_id_score.shape[0]+tag_id_score.shape[0]+cast_id_score.shape[0] == len(id_score))

# 对一个user而言，取活跃的天数的观看的视频列表里频数最大的三个，计算target encoding 的权重平均值(同一天可以看不同的视频)
playback_gb["father_id_score"] = playback_gb.father_id_list.apply(get_id_score, args = (id_score,))
playback_gb["cast_id_score"] = playback_gb.cast_list.apply(get_id_score, args = (id_score,))
playback_gb["tag_score"] = playback_gb.tag_list.apply(get_id_score, args = (id_score,))



# 用户静态信息
portrait = pd.read_csv(input_dir + "user_portrait_data.csv", dtype={"territory_code": str})
portrait = portrait.drop_duplicates()
# portrait["user_id"].value_counts()
previous_shape = total_data_with_all_seq.shape[0]
portrait = pd.merge(total_data_with_all_seq[["user_id", "label"]], portrait, how="left", on="user_id")
print(previous_shape == portrait.shape[0])

df = portrait.loc[(portrait.label >= 0) & (portrait.territory_code.notna()), ["territory_code", "label"]]
territory_score = target_encoding("territory_code", df)

n1 = len(id_score)
id_score.update({x[1]: x[5] for x in territory_score.itertuples()})
print(n1 + territory_score.shape[0] == len(id_score))

portrait["territory_score"] = portrait.territory_code.apply(lambda x: id_score.get(x, 0) if isinstance(x, str) else np.nan)

portrait["device_ram"] = portrait.device_ram.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
portrait["device_rom"] = portrait.device_rom.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)

# interact_prefer特征
interact = pd.read_csv(input_dir + "user_interaction_data.csv")
print(interact.interact_type.min(), interact.interact_type.max())

interact_gb = interact.groupby("user_id").agg(
    interact_type=("interact_type", list)
).reset_index()


def get_interact_prefer(interact_type):
    x = sorted(interact_type)
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x)}
    x_max = max(x_count.values())
    res = [round(x_count.get(i, 0)/x_max, 4) for i in range(1, 12)]
    return res
interact_gb["interact_prefer"] = interact_gb.interact_type.apply(get_interact_prefer)


previous_shape = total_data_with_all_seq.shape[0]

playback_gb_selected = playback_gb[["user_id", "start_date", "end_date", "label", "father_id_score",
                                    "cast_id_score", "tag_score", "duration_prefer"]]
total_data_with_all_seq = total_data_with_all_seq.\
    merge(playback_gb_selected,how = "left", on = ["user_id", "start_date", "end_date", "label"])
total_data_with_all_seq = total_data_with_all_seq.\
    merge(portrait.drop("territory_code", axis = 1), how="left", on=["user_id", "label"])

total_data_with_all_seq= total_data_with_all_seq.\
    merge(interact_gb[["user_id", "interact_prefer"]], on="user_id", how="left")

print(previous_shape == total_data_with_all_seq.shape[0])



#%%
norm_cols = ["father_id_score", "cast_id_score", "tag_score",
            "device_type", "device_ram", "device_rom", "sex",
            "age", "education", "occupation_status", "territory_score"]
for col in norm_cols:
    mean = total_data_with_all_seq[col].mean()
    std = total_data_with_all_seq[col].std()
    total_data_with_all_seq[col] = (total_data_with_all_seq[col] - mean) / std



selected_span = 60
# padding 与计算长度
def padding(x):
    if isinstance(x, list):
        length = len(x)
        if length < 60:
            x.extend([-1]*(60 - length))
        return [x, length]
    else:
        return x
seq_columns_name = ["launch_seq", "launch_type_seq", "playtime_seq", "video_num_seq"]
for name in seq_columns_name:
    total_data_with_all_seq[name] = total_data_with_all_seq[name].apply(padding)
total_data_with_all_seq.fillna({
    "duration_prefer": str([0]*16),
    "interact_prefer": str([0]*11),
    "launch_seq":str([[0] + [-1]*(selected_span - 1), 1]), # 最短长度是1，不能是0
    "launch_type_seq":str([[0] + [-1]*(selected_span - 1), 1]),
    "playtime_seq": str([[0] + [-1]*(selected_span - 1), 1]),
    "video_num_seq":str([[0] + [-1]*(selected_span - 1), 1])
}, inplace=True)


total_data_with_all_seq.fillna(0, inplace=True)
# total_data_with_all_seq = total_data_with_all_seq.drop(["launch_date", "launch_type"], axis = 1)


total_data_with_all_seq.loc[total_data_with_all_seq.label >= 0]\
    .to_csv(output_dir + "train_data.txt", sep="\t", index=False)
total_data_with_all_seq.loc[total_data_with_all_seq.label < 0]\
    .to_csv(output_dir + "test_data.txt", sep="\t", index=False)


