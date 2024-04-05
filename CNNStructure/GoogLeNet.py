import torch
import torch.nn as nn


class googLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7,7),
            stride=2,
            padding=3,
        )
        self.Maxpool = nn.MaxPool2d(kernel_size=(3,3),stride=2)
        self.Conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )