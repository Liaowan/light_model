import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets,transforms
from torch.utils.data import dataloader
from model import ResNet18
import torch.nn as nn
from torch import optim

train_path = os.path.join(os.getcwd(),"dataset","flower_data","train")
val_path = os.path.join(os.getcwd(),"dataset","flower_data","val")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("use {} device".format(device))
transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    "val": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            }
train_data = datasets.ImageFolder(train_path,transform["train"])
val_data = datasets.ImageFolder(val_path,transform["val"])

json_path = os.path.join(os.getcwd(),"dataset/data.json")
assert os.path.exists(json_path), "{} is not exist".format(json_path)
with open(json_path) as file:
    class_dict = json.load(file)

writer = SummaryWriter("logs")

train_dataloader = dataloader.DataLoader(train_data,batch_size=4,num_workers=0,shuffle=True)
val_dataloader = dataloader.DataLoader(val_data,batch_size=1,num_workers=0,shuffle=True)

net = ResNet18().cuda(device)
loss = nn.CrossEntropyLoss()
loss = loss.cuda(device)
optim = optim.Adam(net.parameters(),lr=0.01)

for data in val_dataloader:
    optim.zero_grad()
    img,label = data
    img = img.cuda(device)
    label = label.cuda(device)
    output = net(img)
    loss_fn = loss(output,label)
    print(loss_fn)
    loss_fn.backward()
    optim.step()

