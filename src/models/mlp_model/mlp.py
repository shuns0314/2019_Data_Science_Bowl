import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


COLUMNS_NUM = 512
EPOCH_NUM = 30

def mlp(train: pd.DataFrame, val: pd.DataFrame):
    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=64,
        shuffle=True,
        num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        dataset=val,
        batch_size=64,
        shuffle=False,
        num_workers=2)
    
    net = MLPNet()

    for epoch in range(EPOCH_NUM):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        # ======== train_mode ======
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.view(-1, 28*28*1).to(device), labels.to(device)
            optimizer.zero_grad()  # 勾配リセット
            outputs = net(images)  # 順伝播の計算
            loss = criterion(outputs, labels)  # lossの計算
            train_loss += loss.item()  # train_loss に結果を蓄積
            acc = (outputs.max(1)[1] == labels).sum()  #  予測とラベルが合っている数の合計
            train_acc += acc.item()  # train_acc に結果を蓄積
            loss.backward()  # 逆伝播の計算        
            optimizer.step()  # 重みの更新
            avg_train_loss = train_loss / len(train_loader.dataset)  # lossの平均を計算
            avg_train_acc = train_acc / len(train_loader.dataset)  # accの平均を計算
        
        # ======== valid_mode ======
        net.eval()
        with torch.no_grad():  # 必要のない計算を停止
        for images, labels in valid_loader:        
            images, labels = images.view(-1, 28*28*1).to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            acc = (outputs.max(1)[1] == labels).sum()
            val_acc += acc.item()
        avg_val_loss = val_loss / len(valid_loader.dataset)
        avg_val_acc = val_acc / len(valid_loader.dataset)


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(COLUMNS_NUM, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
