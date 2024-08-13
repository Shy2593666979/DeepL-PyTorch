from torch import nn
import torch
import torchvision
from PIL import Image

data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

class_dict = {}
for goods, idx in data.class_to_idx.items():
    class_dict[idx] = goods

image_path = "../imgs/dog2.png"
image = Image.open(image_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
image = transform(image)

# 搭建神经网络
class TrainModel(nn.Module):
    def __init__(self):
        super(TrainModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, input):
        input = self.model(input)
        return input

model = torch.load("../save_model/train_model.pth", map_location=torch.device("cpu"), weights_only=False)
model.eval()

image = torch.reshape(image, (1, 3, 32, 32))
with torch.no_grad():
    output = model(image)
    
idx = output.argmax(1).item()

print("神经网络模型识别 {}文件 属于 {} 类别".format(image_path, class_dict[idx]))