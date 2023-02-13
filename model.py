import torch
from torch import nn


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*54*54,1000),
            nn.Linear(1000,5),
            nn.Softmax(dim=1),
        )

    def forward(self,x):
        x = self.conv1(x)      # 16*64*54*54

        x = self.conv2(x) + x  # 16*64*54*54
        x = self.conv2(x) + x  # 16*64*54*54
        x = self.conv2(x) + x  # 16*64*54*54
        x = self.conv2(x) + x  # 16*64*54*54
        x = self.conv2(x) + x  # 16*64*54*54

        out = self.conv3(x)

        return out


if __name__ == '__main__':
    net = ResNet18()
    input = torch.zeros([16,3,224,224])
    output = net(input)
    print(output)

