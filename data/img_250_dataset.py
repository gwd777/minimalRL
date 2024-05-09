import argparse
import os
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)  # 128 + 12 = 140
    parser.add_argument('--drop_rate', type=float, default=0.03)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-6)       # 这个参数影响挺大, default=1e-4 Current Learning Rate: [0.0001]
    parser.add_argument('--epochs', type=int, default=100)      # 默认310
    parser.add_argument('--weight_decay', type=float, default=1e-4)     # 0.0001
    parser.add_argument('--data_path', type=str, default='cifa_path')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--print_intervals', type=int, default=100)
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=2.)
    parser.add_argument('--checkpoint_path', type=str, default='./best_model', help='model checkpoints path')
    parser.add_argument('--checkpoints', type=str, default='checkpoint_model_best.pth')    # 'checkpoint_model_best.pth'
    return parser.parse_args()

def load_dataset(args=None):
    data_folder = 'F:\dl_workspace\data'  # 假设图片都存储在 'data_folder' 目录下
    images = []
    labels = []
    for filename in os.listdir(data_folder):  # 遍历目录中的所有文件
        if filename.endswith('.png'):  # 确保只处理 PNG 图片
            file_path = os.path.join(data_folder, filename)
            img = cv2.imread(file_path)
            img = np.transpose(img, (2, 0, 1))
            img = img.astype('float32')
            img /= 255.0
            images.append(img)
            # 根据文件名构造标签
            if filename.startswith('0_'):
                labels.append(0)  # 负样本
            elif filename.startswith('1_'):
                labels.append(1)  # 正样本

    X = np.array(images)
    y = np.array(labels)

    print('X:', len(X), '/ y:', len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将数据转换为PyTorch的张量
    X_train_tensor = torch.tensor(X_train).to(torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test).to(torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    print("________train_dataset / test_dataset________>", train_dataset.__len__(), test_dataset.__len__())

    # 创建DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, train_dataset, test_loader, test_dataset