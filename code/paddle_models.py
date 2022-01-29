import paddle
from paddle.io import DataLoader, Dataset
import pandas as pd
import numpy as np


class CoggleDataset(Dataset):
    def __init__(self, df):
        super(CoggleDataset, self).__init__()
        self.df = df
        # self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label', 'launch_seq', 'playtime_seq',
        #                                                  'duration_prefer', 'interact_prefer']))
        self.feat_col = ['tag_score', 'father_id_score', 'occupation_status', 'territory_score', 'sex', 'device_ram',
                         'device_type', 'device_rom', 'age', 'cast_id_score', 'education']
        self.df_feat = self.df[self.feat_col]

    def __getitem__(self, index):
        launch_type_seq = self.df['launch_type_seq'].iloc[index]
        playtime_seq = self.df['playtime_seq'].iloc[index]
        duration_prefer = self.df['duration_prefer'].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]

        feat = self.df_feat.iloc[index].values.astype(np.float32)

        launch_type_seq = paddle.to_tensor(launch_type_seq).astype(paddle.float32)
        playtime_seq = paddle.to_tensor(playtime_seq).astype(paddle.float32)
        duration_prefer = paddle.to_tensor(duration_prefer).astype(paddle.float32)
        interact_prefer = paddle.to_tensor(interact_prefer).astype(paddle.float32)
        feat = paddle.to_tensor(feat).astype(paddle.float32)

        label = paddle.to_tensor(self.df['label'].iloc[index]).astype(paddle.float32)
        return launch_type_seq, playtime_seq, duration_prefer, interact_prefer, feat, label

    def __len__(self):
        return len(self.df)
class CoggleDataset_7Days(Dataset):
    def __init__(self, df):
        super(CoggleDataset_7Days, self).__init__()
        self.df = df
        # self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label', "label_seq", 'launch_seq', 'playtime_seq',
        #                                                  'duration_prefer', 'interact_prefer']))

        self.feat_col = ['tag_score', 'father_id_score', 'occupation_status', 'territory_score', 'sex', 'device_ram',
                          'device_type', 'device_rom', 'age', 'cast_id_score', 'education']
        self.df_feat = self.df[self.feat_col]

    def __getitem__(self, index):
        launch_type_seq = self.df['launch_type_seq'].iloc[index]
        playtime_seq = self.df['playtime_seq'].iloc[index]
        duration_prefer = self.df['duration_prefer'].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]

        feat = self.df_feat.iloc[index].values.astype(np.float32)

        launch_type_seq = paddle.to_tensor(launch_type_seq).astype(paddle.float32)
        playtime_seq = paddle.to_tensor(playtime_seq).astype(paddle.float32)
        duration_prefer = paddle.to_tensor(duration_prefer).astype(paddle.float32)
        interact_prefer = paddle.to_tensor(interact_prefer).astype(paddle.float32)
        feat = paddle.to_tensor(feat).astype(paddle.float32)

        label_seq = paddle.to_tensor(self.df['label_seq'].iloc[index]).astype(paddle.float32)
        return launch_type_seq, playtime_seq, duration_prefer, interact_prefer, feat, label_seq

    def __len__(self):
        return len(self.df)
class Dataset_Sliding_Window(Dataset):
    def __init__(self, df):
        super(Dataset, self).__init__()
        self.df = df
        self.feat_col = ['tag_score', 'father_id_score', 'occupation_status', 'territory_score', 'sex', 'device_ram',
                         'device_type', 'device_rom', 'age', 'cast_id_score', 'education']
        self.df_feat = self.df[self.feat_col]
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        launch_type_seq, length_launch = self.df['launch_type_seq'].iloc[index]
        playtime_seq, length_playtime = self.df['playtime_seq'].iloc[index]
        duration_prefer = self.df['duration_prefer'].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]

        feat = self.df_feat.iloc[index].values.astype(np.float32)

        launch_type_seq = paddle.to_tensor(launch_type_seq).astype(paddle.float32)
        length_launch = paddle.to_tensor(length_launch).astype(paddle.int64)
        playtime_seq = paddle.to_tensor(playtime_seq).astype(paddle.float32)
        length_playtime = paddle.to_tensor(length_playtime).astype(paddle.int64)
        duration_prefer = paddle.to_tensor(duration_prefer).astype(paddle.float32)
        interact_prefer = paddle.to_tensor(interact_prefer).astype(paddle.float32)
        feat = paddle.to_tensor(feat).astype(paddle.float32)

        label_seq = paddle.to_tensor(self.df['label_seq'].iloc[index]).astype(paddle.float32)
        return launch_type_seq, length_launch, playtime_seq, length_playtime, duration_prefer, interact_prefer, feat, label_seq

class CoggleDataset_5seq(Dataset):
    def __init__(self, df):
        super(CoggleDataset_5seq, self).__init__()
        self.df = df
        # self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label', "label_seq", 'launch_seq', 'playtime_seq',
        #                                                  'duration_prefer', 'interact_prefer']))

        self.feat_col = ['tag_score', 'father_id_score', 'occupation_status', 'territory_score', 'sex', 'device_ram',
                          'device_type', 'device_rom', 'age', 'cast_id_score', 'education']
        self.df_feat = self.df[self.feat_col]

    def __getitem__(self, index):
        launch_seq = self.df["launch_seq"].iloc[index]
        launch_type_seq = self.df['launch_type_seq'].iloc[index]
        playtime_seq = self.df['playtime_seq'].iloc[index]
        video_num_seq = self.df["video_num_seq"].iloc[index]
        interaction_num_seq = self.df["interaction_num_seq"].iloc[index]

        duration_prefer = self.df['duration_prefer'].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]

        feat = self.df_feat.iloc[index].values.astype(np.float32)

        launch_seq = paddle.to_tensor(launch_seq).astype(paddle.float32)
        launch_type_seq = paddle.to_tensor(launch_type_seq).astype(paddle.float32)
        playtime_seq = paddle.to_tensor(playtime_seq).astype(paddle.float32)
        video_num_seq = paddle.to_tensor(video_num_seq).astype(paddle.float32)
        interaction_num_seq =  paddle.to_tensor(interaction_num_seq).astype(paddle.float32)

        duration_prefer = paddle.to_tensor(duration_prefer).astype(paddle.float32)
        interact_prefer = paddle.to_tensor(interact_prefer).astype(paddle.float32)
        feat = paddle.to_tensor(feat).astype(paddle.float32)

        label_seq = paddle.to_tensor(self.df['label_seq'].iloc[index]).astype(paddle.float32)
        return launch_seq, launch_type_seq, playtime_seq, video_num_seq, interaction_num_seq, \
               duration_prefer, interact_prefer, feat, label_seq

    def __len__(self):
        return len(self.df)
class CoggleModel_bl(paddle.nn.Layer):
    def __init__(self):
        super(CoggleModel_bl, self).__init__()

        self.launch_seq_gru = paddle.nn.GRU(1, 32)
        self.playtime_seq_gru = paddle.nn.GRU(1, 32)
        self.fc1 = paddle.nn.Linear(102, 64)
        self.fc2 = paddle.nn.Linear(64, 1)



    def forward(self, launch_seq, playtime_seq, duration_prefer, interact_prefer, feat):
        launch_seq = launch_seq.reshape((-1, 32, 1))
        playtime_seq = playtime_seq.reshape((-1, 32, 1))

        launch_seq_feat = self.launch_seq_gru(launch_seq)[0][:, :, 0]
        playtime_seq_feat = self.playtime_seq_gru(playtime_seq)[0][:, :, 0]

        all_feat = paddle.concat([launch_seq_feat, playtime_seq_feat, duration_prefer, interact_prefer, feat], 1)
        all_feat_fc = self.fc2(self.fc1(all_feat))

        return all_feat_fc
class CoggleModel_deeper(paddle.nn.Layer):
    def __init__(self, fc_input_length = 102, feature_seq_length = 32):
        super(CoggleModel_deeper, self).__init__()

        self.drop_rate = 0.2
        self.fc_input_length = fc_input_length
        self.feature_seq_length = feature_seq_length
        self.launch_seq_gru = paddle.nn.GRU(1, feature_seq_length)
        self.playtime_seq_gru = paddle.nn.GRU(1, feature_seq_length)
        self.fc = paddle.nn.Sequential(paddle.nn.Linear(self.fc_input_length, 256),
                                       paddle.nn.GELU(),
                                       paddle.nn.LayerNorm(256),
                                       paddle.nn.Dropout(self.drop_rate),
                                       paddle.nn.Linear(256, self.fc_input_length),
                                       paddle.nn.GELU(),
                                       paddle.nn.Dropout(self.drop_rate))
        self.classifier = paddle.nn.Sequential(paddle.nn.LayerNorm(self.fc_input_length),
                                               paddle.nn.Linear(self.fc_input_length, 7))
        # self.fc = paddle.nn.Sequential(paddle.nn.Linear(102, 64),
        #                                paddle.nn.Linear(64, 1))

    def forward(self, launch_seq, playtime_seq, duration_prefer, interact_prefer, feat):
        # launch_seq = launch_seq.unsqueeze(2)
        # playtime_seq = playtime_seq.unsqueeze(2)
        launch_seq = launch_seq.reshape((-1, self.feature_seq_length, 1))
        playtime_seq = playtime_seq.reshape((-1, self.feature_seq_length, 1))

        launch_seq_feat = self.launch_seq_gru(launch_seq)[0][:, :, 0]
        playtime_seq_feat = self.playtime_seq_gru(playtime_seq)[0][:, :, 0]

        all_feat = paddle.concat([launch_seq_feat, playtime_seq_feat, duration_prefer, interact_prefer, feat], 1)
        all_feat_fc = self.classifier(self.fc(all_feat) + all_feat)
        # all_feat_fc = self.fc2(self.fc1(all_feat))
        return all_feat_fc
class CoggleModel_bl_7_days(paddle.nn.Layer):
    def __init__(self, fc_input_length = 102, feature_seq_length = 32):
        super(CoggleModel_bl_7_days, self).__init__()

        self.launch_seq_gru = paddle.nn.GRU(1, 32)
        self.playtime_seq_gru = paddle.nn.GRU(1, 32)
        self.fc_input_length = fc_input_length
        self.feature_seq_length = feature_seq_length
        # self.fc1 = paddle.nn.Linear(102, 64)
        #
        # self.fc2 = paddle.nn.Linear(64, 7)
        self.fc = paddle.nn.Sequential(paddle.nn.Linear(self.fc_input_length, 64),
                                       paddle.nn.Linear(64, 7))
    def forward(self, launch_seq, playtime_seq, duration_prefer, interact_prefer, feat):
        launch_seq = launch_seq.reshape((-1, self.feature_seq_length, 1))
        playtime_seq = playtime_seq.reshape((-1, self.feature_seq_length, 1))

        launch_seq_feat = self.launch_seq_gru(launch_seq)[0][:, :, 0]
        playtime_seq_feat = self.playtime_seq_gru(playtime_seq)[0][:, :, 0]

        all_feat = paddle.concat([launch_seq_feat, playtime_seq_feat, duration_prefer, interact_prefer, feat], 1)
        all_feat_fc = self.fc(all_feat)

        return all_feat_fc


class Gru_Encoder(paddle.nn.Layer):
    """ encoder time series """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional="forward"):
        super(Gru_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = paddle.nn.GRU(input_size, hidden_size, num_layers, dropout = 0, direction=bidirectional)

    def forward(self, e_input, h0):
        # output: batch_size * L * hidden_size
        # hn: 1 * batch_size * hidden_size
        output, hn = self.gru(e_input, h0)
        return output, hn
class Gru_Decoder(paddle.nn.Layer):
    """ decoder, input is hidden state of encoder """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional="forward"):
        super(Gru_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = paddle.nn.GRU(input_size, hidden_size, num_layers, dropout=0, direction=bidirectional)

    def forward(self, d_input, h0):
        # output: batch_size * L * hidden_size
        # hn: 1 * batch_size * hidden_size
        output, hn = self.gru(d_input, h0)
        return output, hn
class Gru_Encoder_Decoder(paddle.nn.Layer):

    def __init__(self, e_input_size, e_hidden_size, d_input_size, d_hidden_size, use_gpu = True, device = 0,
                 num_layers=1, bidirectional="forward"):
        super(Gru_Encoder_Decoder, self).__init__()
        self.use_gpu = use_gpu
        self.device = device
        self.num_layers = num_layers
        self.e_hidden_size = e_hidden_size
        self.d_hidden_size = d_hidden_size
        self.encoder = Gru_Encoder(e_input_size, e_hidden_size, num_layers, bidirectional)
        self.decoder = Gru_Decoder(d_input_size, d_hidden_size, num_layers, bidirectional)

    def forward(self, input, target_len):
        """
        :param input:           input data (batch_size * days * e_input_size)
        :param target_len:      time steps of output
        :return:
        """

        batch_size = input.shape[0]
        final_output = self.init_output(batch_size, target_len)
        e_ho = self.init_hidden(batch_size, self.e_hidden_size)
        d_h0 = self.init_hidden(batch_size, self.d_hidden_size)
        # e_output: batch_size * time_steps * e_hidden_size
        e_output, e_hn = self.encoder(input, e_ho)
        decoder_input = paddle.unsqueeze(e_output[:, -1, :], 1)
        for i in range(target_len):
            # decoder_input: batch_size * 1 * e_hidden_size
            d_output, d_hn = self.decoder(e_output[:, -1:, :], d_h0)
            d_h0 = d_hn

            # d_output, d_hn = self.decoder(decoder_input, d_h0)
            # decoder_input, d_h0 = d_output, d_hn # 修改为递归形式

            final_output[:, i, :] = d_output[:, -1, :]
        # final_output: batch_size * target_len * d_hidden_size
        return final_output

    def init_output(self, batch_size, target_len):

        output = paddle.zeros([batch_size, target_len, self.d_hidden_size])

        return output

    def init_hidden(self, batch_size, hidden_size):

        h0 = paddle.zeros([self.num_layers, batch_size, hidden_size])

        return h0
class Model_Single_ED(paddle.nn.Layer):
    def __init__(self, fc_input_length = 7*64 + 38, feature_seq_length = 32, input_seq_dim = 5):
        super(Model_Single_ED, self).__init__()
        self.gru_encoder_decoder1 = Gru_Encoder_Decoder(e_input_size = input_seq_dim,  e_hidden_size = 64,
                                                       d_input_size = 64, d_hidden_size = 64)
        # self.gru_encoder_decoder2 = Gru_Encoder_Decoder(e_input_size = 1, e_hidden_size=64,
        #                                                 d_input_size=64, d_hidden_size=64)
        self.drop_rate = 0.2
        self.fc_input_length = fc_input_length
        self.feature_seq_length = feature_seq_length
        self.embedding = paddle.nn.Embedding(num_embeddings=3, embedding_dim = 4)
        self.fc = paddle.nn.Sequential(paddle.nn.Linear(self.fc_input_length, 256),
                                       paddle.nn.ReLU(),
                                       # paddle.nn.LayerNorm(256),
                                       paddle.nn.Dropout(self.drop_rate))
                                       # paddle.nn.Linear(256, self.fc_input_length),
                                       # paddle.nn.ReLU(),
                                       # paddle.nn.Dropout(self.drop_rate))
        self.classifier = paddle.nn.Sequential(paddle.nn.Linear(256, 7))
    def forward(self, launch_type_seq, playtime_seq, duration_prefer, interact_prefer, feat, target_len = 7):
        launch_type_seq_emb = self.embedding(launch_type_seq.astype(paddle.int64))
        seq_input = paddle.concat([launch_type_seq_emb, playtime_seq.unsqueeze(2)], axis = 2) # [bs, 32, 5]
        seq_output = self.gru_encoder_decoder1(seq_input, target_len) # [bs, 32, 7]
        # seq_output1 = self.gru_encoder_decoder1(launch_type_seq.unsqueeze(2), target_len) # [bs, 7, 64]
        # seq_output2 = self.gru_encoder_decoder2(playtime_seq.unsqueeze(2), target_len) # [bs, 7, 64]
        # seq_output = paddle.concat([seq_output1, seq_output2], axis = 2) #[bs, 7, 128]

        static_info = paddle.concat([duration_prefer, interact_prefer, feat], axis = 1)
        # final_output = None
        # for i in range(target_len):
        #     merge_input = paddle.concat((seq_output[:, i, :], static_info), 1)
        #     cls_input = self.fc(merge_input)
        #     cls_output = self.classifier(cls_input) # [bs, 1]
        #     if final_output is None:
        #         final_output = cls_output
        #     else:
        #         final_output = paddle.concat([final_output, cls_output], axis = 1)
        # return final_output # [bs, 7]
        cls_input = paddle.concat([seq_output.reshape([seq_output.shape[0], -1]), static_info], axis = 1)
        final_output = self.classifier(self.fc(cls_input))
        return final_output


class Model_5seq(paddle.nn.Layer):
    def __init__(self, fc_input_length = 38 + 32*5, feature_seq_length = 32, rnn_hidden_size = 32):
        super(Model_5seq, self).__init__()

        self.fc_input_length = fc_input_length
        self.feature_seq_length = feature_seq_length
        self.rnn_hidden_size = rnn_hidden_size

        self.launch_seq_gru = paddle.nn.GRU(1, self.rnn_hidden_size)
        self.launch_type_seq_gru = paddle.nn.GRU(1, self.rnn_hidden_size)
        self.playtime_seq_gru = paddle.nn.GRU(1, self.rnn_hidden_size)
        self.video_num_seq_gru = paddle.nn.GRU(1, self.rnn_hidden_size)
        self.interaction_num_seq_gru = paddle.nn.GRU(1, self.rnn_hidden_size)


        # self.fc1 = paddle.nn.Linear(102, 64)
        #
        # self.fc2 = paddle.nn.Linear(64, 7)
        self.fc = paddle.nn.Sequential(paddle.nn.Linear(self.fc_input_length, 256),
                                       paddle.nn.Linear(256, 7))
    def forward(self, launch_seq, launch_type_seq, playtime_seq, video_num_seq, interaction_num_seq
                , duration_prefer, interact_prefer, feat):
        launch_seq_feat = self.launch_seq_gru(launch_seq.reshape((-1, self.feature_seq_length, 1)))[0][:, :, 0]
        launch_type_seq_feat = self.launch_type_seq_gru(launch_type_seq.reshape((-1, self.feature_seq_length, 1)))[0][:, :, 0]
        playtime_seq_feat = self.playtime_seq_gru(playtime_seq.reshape((-1, self.feature_seq_length, 1)))[0][:, :, 0]
        video_num_seq_feat = self.video_num_seq_gru(video_num_seq.reshape((-1, self.feature_seq_length, 1)))[0][:, :, 0]
        # interaction_num_seq_feat = self.interaction_num_seq_gru(interaction_num_seq.reshape((-1, self.feature_seq_length, 1)))[0][:, :, 0]


        # all_feat = paddle.concat([launch_type_seq_feat, playtime_seq_feat, duration_prefer, interact_prefer, feat], 1)
        # all_feat = paddle.concat([launch_seq_feat, launch_type_seq_feat, playtime_seq_feat,
        #                           video_num_seq_feat, interaction_num_seq_feat, duration_prefer, interact_prefer,
        #                           feat], axis = 1)

        # interaction_num_seq过于稀疏
        all_feat = paddle.concat([launch_seq_feat, launch_type_seq_feat, playtime_seq_feat,
                                  video_num_seq_feat, duration_prefer, interact_prefer, feat], axis = 1)
        feat_output = self.fc(all_feat)
        return feat_output

class Model_Sliding_Window(paddle.nn.Layer):
    def __init__(self, fc_input_length = 38 + 64*2, feature_seq_length = 60):
        super(Model_Sliding_Window, self).__init__()

        self.launch_seq_gru = paddle.nn.GRU(1, 64)
        self.playtime_seq_gru = paddle.nn.GRU(1, 64)
        self.fc_input_length = fc_input_length
        self.feature_seq_length = feature_seq_length
        # self.fc1 = paddle.nn.Linear(102, 64)
        #
        # self.fc2 = paddle.nn.Linear(64, 7)
        self.fc = paddle.nn.Sequential(paddle.nn.Linear(self.fc_input_length, 256),
                                       paddle.nn.Linear(256, 7))
    def forward(self, launch_seq, L_launch, playtime_seq, L_playtime, duration_prefer, interact_prefer, feat):

        L_launch, L_playtime = L_launch.squeeze(1), L_playtime.squeeze(1)
        # print(paddle.sum(~(paddle.sum(launch_seq != -1, axis = 1) == L_launch)).item(), " ",
        #       paddle.sum(~(paddle.sum(playtime_seq != -1, axis = 1) == L_playtime)).item())

        launch_seq = launch_seq.reshape((-1, self.feature_seq_length, 1))
        playtime_seq = playtime_seq.reshape((-1, self.feature_seq_length, 1))

        launch_seq_output = self.launch_seq_gru(launch_seq, sequence_length = L_launch)[0] # 只能保证length之后的状态不更新
        playtime_seq_output = self.playtime_seq_gru(playtime_seq, sequence_length = L_playtime)[0]
        launch_seq_feat = None
        playtime_seq_feat = None
        for i in range(L_launch.shape[0]): # 并没有看懂paddle.gather_nd的用法
            if launch_seq_feat is None:
                launch_seq_feat = launch_seq_output[i, L_launch[i] - 1, :].unsqueeze(0)
                playtime_seq_feat = playtime_seq_output[i, L_playtime[i] - 1, :].unsqueeze(0)
            else:
                launch_seq_feat = paddle.concat([launch_seq_feat, launch_seq_output[i, L_launch[i] - 1, :].unsqueeze(0)], axis = 0)
                playtime_seq_feat = paddle.concat([playtime_seq_feat, playtime_seq_output[i, L_playtime[i] - 1, :].unsqueeze(0)], axis = 0)

        all_feat = paddle.concat([launch_seq_feat, playtime_seq_feat, duration_prefer, interact_prefer, feat], 1)
        all_feat_fc = self.fc(all_feat)

        return all_feat_fc





