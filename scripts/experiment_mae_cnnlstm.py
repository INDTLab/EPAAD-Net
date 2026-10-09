
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


# from torch.nn.parallel import DataParallel


# 定义时间序列数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data, mask_ratio1, mask_ratio2):
        self.data = data
        self.mask_ratio1 = mask_ratio1
        self.mask_ratio2 = mask_ratio2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 随机遮蔽部分时间序列数据
        masked_data1 = self.data[idx].clone()
        mask_len1 = int(len(masked_data1) * self.mask_ratio1)
        mask_start1 = torch.randint(0, len(masked_data1) - mask_len1, (1,))
        masked_data1[mask_start1:mask_start1 + mask_len1] = 0

        masked_data2 = self.data[idx].clone()
        mask_len2 = int(len(masked_data2) * self.mask_ratio2)
        mask_start2 = torch.randint(0, len(masked_data2) - mask_len2, (1,))
        masked_data2[mask_start2:mask_start2 + mask_len1] = 0

        return masked_data1, masked_data2, self.data[idx]


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


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout=0.2):
        super(CNNLSTM, self).__init__()

        # CNN
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=2, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, input_size)
        self.BN1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(256, input_size)
        self.BN2 = nn.BatchNorm1d(input_size)
        self.relu2 = nn.LeakyReLU()

        # AE
        self.autoencoder1 = Autoencoder(input_size, input_size * 2, input_size)
        self.autoencoder2 = Autoencoder(input_size, input_size * 2, input_size)

    def forward(self, masked_data1, masked_data2):
        # print(masked_data1.shape, masked_data2.shape)

        masked_data1 = self.autoencoder1(masked_data1)
        masked_data2 = self.autoencoder2(masked_data2)

        masked_data1 = masked_data1.permute(1, 0, 2)
        masked_data2 = masked_data2.permute(1, 0, 2)
        N = masked_data1.size(0)
        # print('N', N)
        # print('输入LSTM', x.shape)
        x_lstm = self.lstm(masked_data1)[0]
        # print('lstm处理后', x_lstm.shape)
        x_lstm = self.relu1(self.BN1(self.fc1(x_lstm)))
        # print('调整大小之后', x_lstm.shape)
        x_lstm = x_lstm.mean(1)
        # print('第一个维度平均', x_lstm.shape)
        x_lstm = x_lstm.unsqueeze(1)
        # print('增加一个维度', x_lstm.shape)
        # print('输出LSTM', x_lstm.shape)

        # print('*****************************')
        # print('输入CNN', x.shape)
        x_cnn = self.conv_block1(masked_data2)
        # print('经过第一个卷积块', x_cnn.shape)
        x_cnn = self.conv_block2(x_cnn)
        # print('经过第二个卷积块', x_cnn.shape)
        x_cnn = self.conv_block3(x_cnn)
        # print('经过第三个卷积块', x_cnn.shape)
        x_cnn = torch.mean(x_cnn, 2)
        # print('第二个维度平均', x_cnn.shape)
        x_cnn = self.relu2(self.BN2(self.fc2(x_cnn)))
        # print('调整大小', x_cnn.shape)
        x_cnn = x_cnn.unsqueeze(1)
        # print('增加一个维度', x_cnn.shape)
        # print('输出CNN', x_cnn.shape)

        x = torch.cat([x_cnn, x_lstm], dim=1)
        x = torch.mean(x, 1)
        x = x.unsqueeze(1)
        # print('CNNLSTM拼接后平均：', x.shape)

        x = x.permute(1, 0, 2)
        x = x.repeat(32, 1, 1)
        # print('调整', x.shape)

        return x


# 训练模型
def train_model(model, train_loader, test_loader, labels, optimizer, criterion, epochs=20):
    f1_list = []
    for epoch in range(epochs):
        for masked_data1, masked_data2, target_data in train_loader:
            model.train()
            optimizer.zero_grad()
            output = model(masked_data1, masked_data2)
            # print('ou', output.shape)
            l1 = criterion(output, target_data)
            loss = torch.mean(l1)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.8f}')

        model.eval()  # 设置模型为评估模式
        train_all_losses = []
        with torch.no_grad():  # 不计算梯度
            for masked_data1, masked_data2, target_data in train_loader:
                output = model(masked_data1, masked_data2)
                loss = criterion(output, target_data)
                train_all_losses.append(loss)

        # 将所有batch的loss值拼接成一个tensor
        train_total_loss = torch.cat(train_all_losses, dim=0)
        train_total_loss = torch.mean(train_total_loss, dim=1)

        test_all_losses = []
        with torch.no_grad():  # 不计算梯度
            for target_data in test_loader:
                output = model(target_data, target_data)
                loss = criterion(output, target_data)
                test_all_losses.append(loss)

        # 将所有batch的loss值拼接成一个tensor
        test_total_loss = torch.cat(test_all_losses, dim=0)
        test_total_loss = torch.mean(test_total_loss, dim=1)

        trainfinal, testFinal = np.mean(train_total_loss.cpu().numpy(), axis=1), np.mean(test_total_loss.cpu().numpy(),
                                                                                         axis=1)
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

train_data = train_data[:28448, :]
test_data = test_data[:28448, :]
labels = labels[:28448, :]
train_data = convert_to_windows(train_data).double()  # (28479, 10, 38)
test_data = convert_to_windows(test_data).double()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_data = train_data.to(device)
test_data = test_data.to(device)
# labels = labels.to(device)

# 创建数据集和数据加载器
train_data = TimeSeriesDataset(train_data, mask_ratio1=0.1, mask_ratio2=0.2)
train_loader = DataLoader(train_data, batch_size=32)
# test_data = TimeSeriesDataset(test_data, mask_ratio=0.2)
test_loader = DataLoader(test_data, batch_size=32)

# 创建模型、优化器和损失函数
#model_AE = Autoencoder(input_size=38, hidden_size=64, output_size=38)
model = CNNLSTM(input_size=38, hidden_size=64, output_size=38)
model.double()

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.MSELoss(reduction='none')
criterion = criterion.to(device)

# def train(model_AE, data_loader, criterion, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for data in data_loader:
#             inputs, _ = data
#             inputs = inputs.view(inputs.size(0), -1)  # 将输入展平
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, inputs)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')

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
