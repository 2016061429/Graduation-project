import torch
from torchvision import datasets, transforms
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from resnet import resnet50 as self_resnet50

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# //////////////////////////////////////////////////////
# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.file_paths = []
        self.labels = []
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            files = os.listdir(class_path)
            self.file_paths.extend([os.path.join(class_path, file) for file in files])
            self.labels.extend([i] * len(files))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        data = np.loadtxt(file_path).reshape(32, 512)
        data = torch.from_numpy(data).float()
        return data, label
    
    def get_info(self, idx):
        """ 返回文件路径和标签 """
        return self.file_paths[idx], self.labels[idx]


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 数据集目录和文件路径
    #'D:\PychramProject\transzero'
    # data_dir = 'data/'
    # C:\Users\ZHY\Desktop\data_txt
    # data_dir = 'D:/PychramProject/transzero/data55-512/'
    data_dir = 'C:/Users/ZHY/Desktop/data_txt/'
    class_names = os.listdir(data_dir)

    # 构建数据集
    dataset = MyDataset(data_dir)

    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    # 创建数据加载器
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # //////////////////////////////////////////////////////

    resnet50 = self_resnet50(num_classes=7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络模型cuda
    resnet50 = resnet50.cuda()

    # loss
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()


    # optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(resnet50.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, )


    # 设置网络训练的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 10

    best_acc = 0
    predictions = []
    targets = []
    probabilities = []
    acc = 0

    for i in range(epoch):
        print("-------第{}轮训练开始-------".format(i + 1))
        resnet50.train()
        # 训练步骤开始
        for data in tqdm(train_dataloader, ncols=100, desc='Train'):
            imgs, targets = data
            if torch.cuda.is_available():
                # 图像cuda；标签cuda
                # 训练集和测试集都要有
                imgs = imgs.cuda()
                targets = targets.cuda()
            imgs = imgs.reshape(-1, 1, 32, 512)
            # imgs1 = torch.cat((imgs, imgs), 1)
            # imgs2 = torch.cat((imgs, imgs1), 1)
            outputs = resnet50(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            # if total_train_step % 100 == 0:
        print("训练次数：{}, Loss: {:.4f}".format(total_train_step, loss.item()))
                # writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试集
        # predictions = []
        # targets = []
        # probabilities = []
        #
        resnet50.eval()
        total_test_loss = 0
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0

            for data in test_dataloader:
                imgs, targets = data
                if torch.cuda.is_available():
                    # 图像cuda；标签cuda
                    # 训练集和测试集都要有
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                imgs = imgs.reshape(-1, 1, 32, 512)
                # imgs1 = torch.cat((imgs, imgs), 1)
                # imgs2 = torch.cat((imgs, imgs1), 1)
                outputs = resnet50(imgs)
                #
                # intermediate_layer = resnet50.layer4[-1].conv3  # Modify this line to select the desired intermediate layer
                # visualize_features(outputs)
                #
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.tolist())
                #
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                total_test_step += 1

                # logits = resnet50(imgs)
                logits = outputs
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, targets).float().sum().item()
                total_correct += correct
                total_num += imgs.size(0)
                #
                softmax = nn.Softmax(dim=1)
                probs = softmax(outputs)
                probabilities.extend(probs.tolist())


                # if total_test_step % 100 == 0:
            print("测试次数：{}，Loss：{:.4f}".format(total_test_step, total_test_loss))
            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)
            #
            # print("定性预测结果：", predictions)
            # print("定量预测结果：", probabilities)
            # 保存最优模型
            if acc > best_acc:
                best_acc = acc
                torch.save(resnet50.state_dict(), 'best_model.pt')

    # 将结果写入文件
    result_file = 'result.txt'
    with open(result_file, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("定性预测结果：\n")
        for pred in predictions:
            f.write(f"{pred}+','")
        f.write("定量预测结果：\n")
        for prob in probabilities:
            f.write(f"{prob}+','")