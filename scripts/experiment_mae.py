
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from src.pot import *
from pprint import pprint
#from torch.nn.parallel import DataParallel


# 定义时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data, mask_ratio=0.2):
        self.data = data
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 随机遮蔽部分时间序列数据
        masked_data = self.data[idx].clone()
        mask_len = int(len(masked_data) * self.mask_ratio)
        mask_start = torch.randint(0, len(masked_data) - mask_len, (1,))
        masked_data[mask_start:mask_start + mask_len] = 0

        return masked_data, self.data[idx]


# 定义编码器-解码器模型
class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, output_size)

    def forward(self, x):
        # 编码器
        _, (h_n, _) = self.encoder(x)
        # 解码器
        output, _ = self.decoder(h_n)
        output = output.repeat_interleave(x.size(0), dim=0)
        return output


# 训练模型
def train_model(model, train_loader, test_loader, labels, optimizer, criterion, epochs=50):
    f1_list = []
    for epoch in range(epochs):
        for masked_data, target_data in train_loader:
            model.train()
            optimizer.zero_grad()
            output = model(masked_data)
            # print('ou', output.shape)
            l1 = criterion(output, target_data)
            loss = torch.mean(l1)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.8f}')

        model.eval()  # 设置模型为评估模式
        train_all_losses = []
        with torch.no_grad():  # 不计算梯度
            for masked_data, target_data in train_loader:
                output = model(masked_data)
                loss = criterion(output, target_data)
                train_all_losses.append(loss)

        # 将所有batch的loss值拼接成一个tensor
        train_total_loss = torch.cat(train_all_losses, dim=0)
        train_total_loss = torch.mean(train_total_loss, dim=1)

        test_all_losses = []
        with torch.no_grad():  # 不计算梯度
            for masked_data, target_data in test_loader:
                output = model(masked_data)
                loss = criterion(output, target_data)
                test_all_losses.append(loss)

        # 将所有batch的loss值拼接成一个tensor
        test_total_loss = torch.cat(test_all_losses, dim=0)
        test_total_loss = torch.mean(test_total_loss, dim=1)

        trainfinal, testFinal = np.mean(train_total_loss.cpu().numpy(), axis=1), np.mean(test_total_loss.cpu().numpy(), axis=1)
        labFinal = (np.sum(labels, axis=1) >= 1) + 0
        result, _ = pot_eval(trainfinal, testFinal, labFinal)
        print('f1', result['f1'], ' ', 'pre', result['precision'], ' ', 'recall', result['recall'])

        f1_list.extend([result['f1']])
        maxf1 = 0
        for i in range(len(f1_list)):
            if f1_list[i] > maxf1:
                maxf1 = f1_list[i]
                print(i, end=' ')
        print('maxf1', maxf1)

# 测试模型
def test_model(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    all_losses = []
    with torch.no_grad():  # 不计算梯度
        for masked_data, target_data in test_loader:
            output = model(masked_data)
            loss = criterion(output, target_data)
            all_losses.append(loss)

    # 将所有batch的loss值拼接成一个tensor
    total_loss = torch.cat(all_losses, dim=0)
    total_loss = torch.mean(total_loss, dim=1)
    return total_loss


def convert_to_windows(data):
    windows = []
    w_size = 10

    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w)

    return torch.stack(windows)


# 加载时间序列数据
train_data = torch.tensor(np.load(f'processed/SMD/machine-1-1_train.npy'))  # (28479, 38)
test_data = torch.tensor(np.load(f'processed/SMD/machine-1-1_test.npy'))
labels = np.load(f'processed/SMD/machine-1-1_labels.npy')
# train_data = torch.tensor(np.load(f'processed/SWaT/train.npy'))  # (28479, 38)
# test_data = torch.tensor(np.load(f'processed/SWaT/test.npy'))
# labels = np.load(f'processed/SWaT/labels.npy')
# train_data = torch.tensor(np.load(f'processed/MSL/C-1_train.npy'))  # (28479, 38)
# test_data = torch.tensor(np.load(f'processed/MSL/C-1_test.npy'))
# labels = np.load(f'processed/MSL/C-1_labels.npy')

train_data = convert_to_windows(train_data).double()  # (28479, 10, 38)
test_data = convert_to_windows(test_data).double()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = train_data.to(device)
test_data = test_data.to(device)
#labels = labels.to(device)

# 创建数据集和数据加载器
train_data = TimeSeriesDataset(train_data)
train_loader = DataLoader(train_data, batch_size=32)
test_data = TimeSeriesDataset(test_data)
test_loader = DataLoader(test_data, batch_size=32)


# 创建模型、优化器和损失函数
model = EncoderDecoder(input_size=38, hidden_size=64, output_size=38)
model.double()
#model = DataParallel(model)
model = model.to(device)
# optimizer = optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.MSELoss(reduction='none')
criterion = criterion.to(device)

# 训练模型
train_model(model, train_loader, test_loader, labels, optimizer, criterion)
test_loss = test_model(model, test_loader, criterion)
print('testloss', test_loss.shape)

train_loss = test_model(model, train_loader, criterion)
print('trainloss', train_loss.shape)

losstrainfinal, losstestFinal = np.mean(train_loss.cpu().numpy(), axis=1), np.mean(test_loss.cpu().numpy(), axis=1)
labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
result, _ = pot_eval(losstrainfinal, losstestFinal, labelsFinal)

pprint(result)


# # 定义 MAE 模型
# class MAE(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded



# # 使用训练好的模型进行异常检测
# def detect_anomaly(model, test_data, threshold=0.1):
#     """
#     使用训练好的模型进行异常检测。
#
#     Args:
#         model: 训练好的 EncoderDecoder 模型。
#         test_data: 测试数据，形状为 (N, 1)。
#         threshold: 异常检测阈值。
#
#     Returns:
#         anomaly_indices: 异常点的索引列表。
#     """
#
#     # 将测试数据分成多个时间段，每个时间段长度为模型输入长度
#     input_len = model.encoder.input_size
#     print(input_len)
#     test_data_segments = [test_data[i:i + input_len] for i in range(0, len(test_data) - input_len + 1)]
#
#     # 使用模型预测测试集
#     predictions = [model(segment) for segment in test_data_segments]
#
#
#     # # 使用模型预测每个时间段的重建结果
#     # reconstructions = [model(segment).squeeze(0) for segment in test_data_segments]
#     #
#     # # 计算每个时间段的重建误差
#     # reconstruction_errors = [torch.mean(torch.abs(segment - reconstruction)) for segment, reconstruction in
#     #                          zip(test_data_segments, reconstructions)]
#     #
#     # # 找出重建误差大于阈值的点
#     # anomaly_indices = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
#
#     return predictions