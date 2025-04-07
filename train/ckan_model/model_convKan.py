import KANConv
import torch
import torch.nn as nn

KAN_Convolutional_Layer = KANConv.KAN_Convolutional_Layer
class KANC_MLP(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=3,
            kernel_size=(3, 3),
            device=device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs=2,
            kernel_size=(3, 3),
            device=device
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(3528, 1024)#in_feature在需要时可以修改 1176为灰度,3528为彩色
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)

        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


