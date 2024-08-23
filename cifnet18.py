import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader as DataLoader 
import numpy as np
""" class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 构建残差网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
     """
class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# 定义数据转换
transform_train = torchvision.transforms.Compose([
    # 原本图像是32*32，先放大成40*40， 在随机裁剪为32*32，实现训练数据的增强
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]
    ),
    Cutout(n_holes=1, length=16),
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道 ： 消除评估结果中的随机性
    torchvision.transforms.Normalize(
        [0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]
    )
])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def main(): 
# 加载数据集
 train_data = torchvision.datasets.CIFAR10(root='D:/python/dataset', train=True, transform=transform_train,
                                          download=False)
 test_data = torchvision.datasets.CIFAR10(root='D:/python/dataset', train=False, transform=transform_test,
                                         download=False)
 train_data_size = len(train_data)
 test_data_size = len(test_data)
 print("训练集的长度为:{}".format(train_data_size))
 print("测试集的长度为:{}".format(test_data_size))
 train_dataloader = DataLoader(train_data, batch_size=64)
 test_dataloader = DataLoader(test_data, batch_size=64)
# 加载预训练的resnet18模型
 model = torchvision.models.resnet18(pretrained=True).to(device)
 model.conv1 = nn.Conv2d(model.conv1.in_channels, 
                        model.conv1.out_channels,
                        3, 1, 1, bias=False).to(device)
 model.maxpool = nn.Identity()
 #model.fc = nn.Linear(model.fc.in_features, 10)
 #model=ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10).to(device)
# 更改最后的全连接层为10分类
 num_ftrs = model.fc.in_features
 model.fc = nn.Linear(num_ftrs, 10).to(device)
 
# 定义损失函数和优化器
 loss_fn = nn.CrossEntropyLoss()
 
 
# 训练模型的代码（省略）
# ...
 checkingpath='D:/python/dataset/checkpoint/resnet18_cifar10.pt'
 if(os.path.exists(checkingpath)):   
    model.load_state_dict(torch.load(checkingpath ))
    model = model.to(device) 
    #model = torch.load(checkingpath,weights_only=False).to(device)
# 在测试集上测试模型的代码（省略）
 total_test_loss = 0
 total_accuracy = 0
 total_test_step = 0
# 记录训练网络的一些参数
# 记录训练次数
 total_train_step = 0
 valid_loss_min = np.Inf
# 记录训练的轮数
 epoch = 200
 lr = 0.1
 for i in range(epoch):
        print("-----第{}轮训练开始-----".format(i+1))
           
        total_accuracy = 0
        if(i%10==0 and i>0):
            
            lr = lr*0.5
            #print("-----第{}轮保存-----".format(i/10))
            #torch.save(model, "D:/python/dataset/cif18"+str(i)+".pth")

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)    
        # 开始训练
        model.train()  # 仅对特殊的网络层有作用
        for data in train_dataloader:
            imgs, targets = data
            imgs=imgs.to(device)
            targets=targets.to(device)
            
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = loss.item()*imgs.size(0)
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            preds=outputs.argmax(1)
            accuracy = ( preds== targets).sum().item() 
            total_accuracy += accuracy

            
            if total_train_step % 100 == 0:
                print("训练集的Loss:{}".format(total_test_loss/imgs.size(0)))
              

                print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
        print("训练集的正确率：{}".format(total_accuracy/train_data_size))   


        # 开始测试
        model.eval()
        total_accuracy = 0    
        total_test_loss=0   
 with torch.no_grad():
  for data in test_dataloader:
                imgs, targets = data
                imgs=imgs.to(device)
                targets=targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()*imgs.size(0)
                # argmax找出每一张预测概率最大的下标
                accuracy = (outputs.argmax(1) == targets).sum().item() 
                total_accuracy += accuracy

 print("测试集的正确率：{}".format(total_accuracy/test_data_size))
 valid_loss = total_test_loss/test_data_size
 if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model, 'D:/python/dataset/checkpoint/resnet18_cifar10.pt')
        valid_loss_min = valid_loss        


if __name__ == '__main__':
    main()