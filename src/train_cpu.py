import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from model import TrainModel
from torch.utils.tensorboard.writer import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(train_data_size)
print(test_data_size)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
train_model = TrainModel()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(train_model.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0

total_test_step = 0

# 循环的轮数
epoch = 10

writer = SummaryWriter("../log_train")

for i in range(epoch):
    print("-----第 {} 轮训练开始-----".format(i+1))
    
    # 训练步骤开始
    train_model.train()
    for data in train_dataloader:
        imgs, targets = data
        output = train_model(imgs)
        loss = loss_fn(output, targets)
        
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        
    # 测试步骤开始
    train_model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            output = train_model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
        
        print("测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size)
        total_test_step += 1

writer.close()