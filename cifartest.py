import torch
import torchvision
import torch.nn as nn
from PIL import Image, ImageDraw,ImageFont
import shutil
import os
import torch.nn.functional as F

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'   
class_id = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '轮船', '卡车']
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道 ： 消除评估结果中的随机性
    torchvision.transforms.Normalize(
        [0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]
    )
])

def enumerate_files(directory):
 
# 列出目录下的所有文件和文件夹
    entries = os.listdir(directory)
    #print (str(entries))
    # 过滤出文件
    wholefilesos=[]
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
    for curf in files:
        wholefilesos.append(os.path.join(directory, curf))

    return wholefilesos
def CopycatdogFile(model):
        
        dataset_test='D:/python/dataset/test/'
        model.eval()
       # font = ImageFont.truetype('simsun.ttc', 36)
        for file_path in enumerate_files(dataset_test):
          
            #print(curfilepath)
            image = Image.open(file_path).convert('RGB')
           
          

            # 网络的输入为32X32
            transform =transform_test#torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),torchvision.transforms.ToTensor()])
            image = transform(image)
            #print(image.shape)

            # [3,32,32] --> [1,3,32,32]
            image = torch.reshape(image,[1, 3, 32, 32]).to(device)
            
            with torch.no_grad():
                output = model(image)
                idx = output.argmax(1)
                reslut = class_id[idx]

           

                  
            caterror=dataset_test+reslut    
            if(not os.path.exists(caterror)):    
                os.mkdir(caterror)
            curfilename=file_path.split(sep='/')[-1]
            finalname=caterror+'/'+ curfilename   
                            #print (caterror)
            shutil.copyfile(file_path,finalname)
           
def main():
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #image_path = 'D:\\python\\dataset\\67.jpg'
    #image_path = 'D:\Data\Learn_Pytorch\cifar10_train\plane.png'
    model = torchvision.models.resnet18(pretrained=True).to(device)
    model.conv1 = nn.Conv2d(model.conv1.in_channels,model.conv1.out_channels,3, 1, 1, bias=False).to(device)
    model.maxpool = nn.Identity()
 #model.fc = nn.Linear(model.fc.in_features, 10)
 #model=ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10).to(device)
#更改最后的全连接层为10分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10).to(device)
 
    #checkingpath='D:/python/dataset/checkpoint/resnet18_cifar10.pt'
    checkingpath='D:/python/dataset/checkpoint/cif18.9139.pth'
    model = torch.load(checkingpath,weights_only=False)

    #model.load_state_dict(torch.load(checkingpath ))
    model = model.to(device)      
    
    # checkpoint = torch.load('D:/python/dataset/checkpoint/ckpt.pth')
    # net = SimpleDLA()
    # net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
        
        
    # net.load_state_dict(checkpoint['net'])

    #model = torchvision.models.resnet18(pretrained=True).to(device)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 10).to(device)
    CopycatdogFile(model)
if __name__ == "__main__":
    main()