## 结果
### ✨ 训练过程图

训练10轮数据，每次训练完一轮在测试集上的正确率逐渐提高：
![1723562482509](https://github.com/user-attachments/assets/ffda4925-e4a2-41d8-8885-cae18d1cf2f7)

### 🎨 测试

最后在test.py文件中测试一下识别准确率怎么样？

![image](https://github.com/user-attachments/assets/bc23e980-c5db-4c91-b688-634b405cfed4)

正确率可达50% 及以上

### 🎁 损失函数图

![1723563587266](https://github.com/user-attachments/assets/ddd6ee40-c9fa-42e6-9270-973a823e3c61)


## 具体介绍

### 📊 数据集

CIFAR-10 数据集是一个广泛用于图像分类任务的数据集。它包含 10 个类别的 60,000 张 32x32 彩色图像，每个类别有 6,000 张图像。其中 50,000 张图像用于训练，10,000 张图像用于测试。

运行代码时，数据集将自动下载到 ../data 目录中。


### 🧠 模型

模型架构在 TrainModel 类中定义，该类应在 model.py 文件中实现。该模型旨在将 CIFAR-10 数据集中的图像分类到 10 个类别之一。


### 🚀 训练

训练过程由主脚本控制。模型使用随机梯度下降（SGD）优化器和交叉熵损失函数进行训练。


要开始训练，只需运行以下脚本：
```bash
python train.py
```
训练过程将记录损失和准确率等指标到 TensorBoard，可以通过以下命令可视化这些指标：


```bash
tensorboard --logdir=../log_train
```
### 🔍 评估

每个 epoch 结束后，模型将在测试数据集上进行评估。评估指标（包括损失和准确率）也会记录到 TensorBoard 中。

### 🏆 结果

最终训练好的模型将保存为根目录下的 train_model.pth 文件。

您可以加载并使用此模型进行推理或进一步训练：

```python
import torch

model = torch.load("train_model.pth")
model.eval()
```
