import copy
import time
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet18, Residual
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import argparse
from MyModule.SEBlock import SEBlock


def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 with optional SEBlock")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_se', action='store_true', help='Use SEBlock if set')

    return parser.parse_args()


def train_val_data_process(batch_size=32):
    # 定义数据集的路径
    ROOT_TRAIN = './dataset/train'

    normalize = transforms.Normalize([0.48607032,0.45353173,0.4160252 ], [0.06886391,0.06542894,0.0667423 ])
    # 定义数据集处理方法变量
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    train_data, val_data = Data.random_split(train_data, [round(0.9*len(train_data)), round(0.1*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader, val_dataloader



def train_model_process(model, train_dataloader, val_dataloader, num_epochs=50, lr=0.001, save_name="undefined"):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    for epoch in range(num_epochs):
        since = time.time()

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc=f"Epoch Train {epoch + 1}/{num_epochs}"):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        for step, (b_x, b_y) in tqdm(enumerate(val_dataloader),total=len(val_dataloader), desc=f"Epoch val {epoch + 1}/{num_epochs}"):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)


            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        time_use = time.time() - since
        print('\n' + """Epoch {} Time {:.4f} m {:.4f}s
                Train Loss: {:.4f} Train Acc: {:.4f}
                Val Loss: {:.4f} Val Acc: {:.4f}
                """.format(epoch + 1, time_use // 60, time_use % 60, 
                           train_loss_all[-1], train_acc_all[-1], 
                           val_loss_all[-1], val_acc_all[-1]))
        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

    folder_path = "./model"
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]) + 1
    torch.save(best_model_wts, f"./model/{save_name}-{file_count}.pth")
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })
    return train_process, f"./model/best_model{file_count}.pth"


def matplot_acc_loss(train_process,png_name="undefined"):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    # plt.show()
    plt.savefig(f"{png_name}.png")


if __name__ == '__main__':
    args = get_args()
    now_parameters = f"epochs:{args.epochs}-batch_size:{args.batch_size}-lr:{args.lr}-SEBlock:{args.use_se}".replace('.', '_')
    attention = SEBlock if args.use_se else None
    model = ResNet18(Residual, attention=attention)
    train_data, val_data = train_val_data_process(batch_size=args.batch_size)
    train_process, src = train_model_process(model, train_data, val_data, num_epochs=args.epochs, lr=args.lr, save_name=now_parameters)
    # python model_train.py --epochs 50 --batch_size 64 --lr 0.0005 --use_se
    png_name = "process_data"+now_parameters
    matplot_acc_loss(train_process,png_name)
    print(f"Finish:{src}")