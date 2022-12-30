import torch.nn as nn


class Flattener(nn.Module):

    def forward(self, batch_size, x):
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)


class CNN(nn.Module):
    def __init__(self, img_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 236, 5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.Conv2d(236, 236, 5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            Flattener(),
            nn.Linear(236*2*2, out_size)
        )

    def forward(self, x):
        return self.model(x)
