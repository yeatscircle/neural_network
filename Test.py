# import numpy as np
#
# arr = np.array([1, 2, 3, 4, 5])
# print(arr)
# print(arr.shape)
# print(type(np.random))
#
# my_array = np.random.random((2,3))
# print(my_array)
# my_array_column_2 = my_array[0, (1,2)]
# print(my_array_column_2)
#
# b = np.ones((25))
# b = b.reshape((5,5))
# print(b)
# print(np.dot(arr,b))
# print(arr@b)
# a = np.arange(0, 100, 10)
# print(a)
#
# # Boolean masking
# # 导入 matplotlib.pyplot 用于数据可视化
# import matplotlib.pyplot as plt
# # 导入 NumPy 库，用于数组操作
# import numpy as np
#
# # 创建一个从 0 到 2π 的等差数列，包含 50 个点
# a = np.linspace(0, 2 * np.pi, 50)
# # 计算 a 中每个点的正弦值
# b = np.sin(a)
#
# # 绘制 a 对 b 的图像，即正弦曲线
# plt.plot(a, b)
#
# # 创建一个布尔掩码，表示 b 中所有非负值
# mask = b >= 0
# # 使用布尔掩码绘制所有非负的正弦值点，用蓝色圆圈 'bo' 表示
# plt.plot(a[mask], b[mask], 'bo')
#
# plt.show()
# # 创建另一个布尔掩码，同时满足 b 非负且 a 小于等于 π/2
# mask = (b >= 0) & (a <= np.pi / 2)
# # 使用布尔掩码绘制同时满足 b 非负且 a 小于等于 π/2 的点，用绿色圆圈 'go' 表示
# plt.plot(a[mask], b[mask], 'go')
#
# # 显示图表
# plt.show()
#
# # Where
# a = np.arange(0, 100, 10)
# b = np.where(a < 50)
# c = np.where(a >= 50)[0]
# print(type(b))
# print(b) # >>>(array([0, 1, 2, 3, 4]),)
# print(c) # >>>[5 6 7 8 9]
#
# import numpy as np
#
# # We will add the vector v to each row of the matrix x,
# # storing the result in the matrix y
# x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# v = np.array([1, 0, 1])
# y = np.empty_like(x)   # Create an empty matrix with the same shape as x
# print(y)
# # Add the vector v to each row of the matrix x with an explicit loop
# for i in range(4):
#     y[i, :] = x[i, :] + v
#
# # Now y is the following
# # [[ 2  2  4]
# #  [ 5  5  7]
# #  [ 8  8 10]
# #  [11 11 13]]
# print(y)
# print(type(y))
#
# tt = np.array("hell0")
# print(type(tt))
#
# A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
# b = np.transpose(np.array([[-3,5,-2]]))
# x = np.linalg.solve(A,b)
# print(x)

# import numpy as np
# import torch
# import random
#
# def set_seed(seed:int):
#     np.random.seed(seed)
#     random.seed(seed)
#     # cpu
#     torch.manual_seed(seed)
#     # gpu
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
#     print(f'set env random_seed:{seed}')

# import os
# import json
# import torch
# import random
# from pathlib import Path
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn as nn
#
#
# class myDataset(Dataset):
#     def __init__(self, data_dir, segment_len=128):
#         self.data_dir = data_dir
#         self.segment_len = segment_len
#
#         # Load the mapping from speaker name to their corresponding id
#         mapping_path = Path(data_dir) / "mapping.json"
#         mapping = json.load(mapping_path.open())
#         self.speaker2id = mapping["speaker2id"]
#
#         # Load metadata of training data
#         metadata_path = Path(data_dir) / "metadata.json"
#         metadata = json.load(open(metadata_path))["speakers"]
#
#         # Get the total number of speaker
#         self.speaker_num = len(metadata.keys())
#         self.data = []
#         for speaker in metadata.keys():
#             for utterances in metadata[speaker]:
#                 self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
#
#         def __len__(self):
#             return len(self.data)
#
#         def __getitem__(self, index):
#             feat_path, speaker = self.data[index]
#             # Load preprocessed mel-spectrogram.
#             mel = torch.load(os.path.join(self.data_dir, feat_path))
#
#             # Segmemt mel-spectrogram into "segment_len" frames.
#             if len(mel) > self.segment_len:
#                 # Randomly get the starting point of the segment.
#                 start = random.randint(0, len(mel) - self.segment_len)
#                 # Get a segment with "segment_len" frames.
#                 mel = torch.FloatTensor(mel[start:start + self.segment_len])
#             else:
#                 mel = torch.FloatTensor(mel)
#             # Turn the speaker id into long for computing loss later.
#             speaker = torch.FloatTensor([speaker]).long()
#             return mel, speaker
#
#         def get_speaker_number(self):
#             return self.speaker_num
#
#
# class Classifier(nn.Module):
#     def __init__(self,d_model=80,n_spks=600,dropout=0.1):
#         super(Classifier,self).__init__()
#
#         # Project the dimension of features from that of input d_model
#         self.prenet = nn.Linear(40,d_model)
#
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, dim_feedforward=256, n_head=2
#         )
#         # self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=2)
#
#         # Project the dimension of features from d_model into speaker nums
#         self.pred_layer = nn.Sequential(
#             nn.Linear(d_model,d_model),
#             nn.ReLU(),
#             nn.Linear(d_model,n_spks)
#         )
#
#     def forward(self,mels):
#         """
#         args:
#         	mels: (batch size, length, 40)
#         return:
#         	out: (batch size, n_spks)
#         """

from PIL import Image
import os

ima = Image.open("C:\\Users\\SN\\Desktop\\1000268201_693b08cb0e.jpg")
        




