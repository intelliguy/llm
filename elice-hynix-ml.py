from elice_utils import EliceUtils
elice_utils = EliceUtils()

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


SEED = 2022
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def preprocess_data(df_train, df_test):
    train_data = {
        "time": df_train.iloc[:, 0].values,
        "X": df_train.iloc[:, 1].values,
        "y": df_train.iloc[:, 2:].values,
    }
    test_data = {
        "time": df_test.iloc[:, 0].values,
        "X": df_test.iloc[:, 1].values,
        "y": df_test.iloc[:, 2:].values,
    }

    # TODO: [지시사항 1-A번] 입력 데이터를 표준화합니다.
    X_mean = np.mean(train_data["X"])
    X_std = np.std(train_data["X"])

    X_train = train_data["X"] - X_mean
    X_train = X_train / X_std

    X_test = test_data["X"] - X_mean
    X_test = X_test / X_std

    # TODO: [지시사항 1-B번] 출력 데이터를 표준화합니다.
    y_mean = np.mean(train_data["y"], axis=0)
    y_std = np.std(train_data["y"], axis=0)

    y_train = train_data["y"] - y_mean
    y_train = y_train / y_std

    y_test = test_data["y"] - y_mean
    y_test = y_test / y_std

    return X_train, X_test, y_train, y_test


def process_time_series_data(X, y, num_sequences, stride=1):
    X_ = []
    for i in range(num_sequences):
        # TODO: [지시사항 2-A번] 지시사항에 나오는 시계열 데이터 변환 원리에 따라
        # X_ array에 sequence로 구분한 데이터를 추가합니다.
        start_idx = i * stride
        end_idx = start_idx + num_sequences
        X_.append(X[start_idx:end_idx])

    # TODO: [지시사항 2-B번] 지시사항에 나오는 시계열 데이터 변환 원리에 따라
    # window size로 구분된 데이터를 만듭니다.
    y_ = []
    for X_seq in X_:
        y_.append(y[X_seq[-1]])

    return X_, y_


class LSTMModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # TODO: [지시사항 3번] LSTM 모델을 만듭니다.
        self.lstm1 = nn.LSTM(num_features, 128)
        self.lstm2 = nn.LSTM(128, 64)
        self.linear = nn.Linear(64, 5)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.linear(x)

        return x


class GRUModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # TODO: [지시사항 4번] GRU 모델을 만듭니다.
        self.gru1 = nn.GRU(num_features, 128)
        self.gru2 = nn.GRU(128, 64)
        self.linear = nn.Linear(64, 5)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.linear(x)

        return x


def plot_prediction(y_test, y_pred):
    start_i, end_i = 10000, 20000
    out_name = ["A", "B", "C", "D", "E"]

    for out_idx, out_name in zip(range(y_pred.shape[-1]), out_name):
        fig = plt.figure(figsize=(10, 5))

        plt.xlabel("Time (picoseconds)")
        plt.ylabel("Values")
        plt.title(f"Model Prediction - {out_name}")

        plt.plot(y_test[start_i:end_i, out_idx], color="b", label="real")
        plt.plot(y_pred[start_i:end_i, out_idx], color="r", label="pred")
        plt.legend(loc="upper right")

        plt.savefig(f"hynix_test_result_{out_name.lower()}.png")
        elice_utils.send_image(f"hynix_test_result_{out_name.lower()}.png")


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y


def train(model, loader, optimizer, criterion, device):
    model.train()
    num_epochs = 10
    steps_per_epoch = 100

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        num_steps = 0

        pbar = tqdm(loader)
        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_steps += 1
            if num_steps == steps_per_epoch:
                break

        print(f"[Epoch {epoch}/{num_epochs}] loss = {epoch_loss / len(loader):.5e}")


def test(model, loader, criterion, device):
    model.eval()
    pred_list = []

    test_loss = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        test_loss += loss.item()
        pred_list.append(pred.detach().cpu().numpy())

    pred_list = np.concatenate(pred_list, axis=0)
    print(f"[Test] loss = {test_loss / len(loader):.5e}")

    return pred_list


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    root_dir = "."
    df_train = pd.read_csv(
        os.path.join(root_dir, "SKHY_train.txt"), delimiter=",", header=0
    )
    df_test = pd.read_csv(
        os.path.join(root_dir, "SKHY_test_answer.txt"), delimiter=",", header=0
    )

    X_train, X_test, y_train, y_test = preprocess_data(df_train, df_test)

    num_sequences = 150
    stride = 1
    X_train2, y_train2 = process_time_series_data(
        X_train, y_train, num_sequences, stride
    )
    X_test2, y_test2 = process_time_series_data(
        X_test, y_test, num_sequences, stride
    )

    train_set = CustomDataset(X=X_train2, y=y_train2)
    test_set = CustomDataset(X=X_test2, y=y_test2)

    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # GRU 모델을 사용할 경우 LSTMModel을 GRUModel로 변경
    num_features = train_set[0][0].shape[-1]
    model = LSTMModel(num_features).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train(model, train_loader, optimizer, criterion, device)

    y_pred = test(model, test_loader, criterion, device)
    print(f"테스트 MSE: {mean_squared_error(y_test2, y_pred):.5E}")
    plot_prediction(y_test2, y_pred)


if __name__ == "__main__":
    main()