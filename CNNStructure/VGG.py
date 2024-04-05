import torch
import torch.nn as nn
from torchvision import transforms as transforms

My_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.AutoAugment(),
    transforms.ToTensor(),
])

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class Vgg(nn.Module):
    def __init__(self, transform: transforms = None, num_classes=1000, in_channels=3):
        super(Vgg, self).__init__()
        self.in_channels = in_channels
        self.transform = transform
        self.conv_layers = self.create_layes(VGG_types["VGG16"])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_layes(self, architecture):
        layes = []
        in_channels = self.in_channels

        for temp in architecture:
            if temp == 'M':
                layes += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layes += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=temp,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(temp),
                    nn.ReLU(),
                ]
                in_channels = temp
        return nn.Sequential(*layes)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Vgg(in_channels=3, num_classes=1000).to(device)
    BATCH_SIZE = 3
    x = torch.randn(3, 3, 224, 224).to(device)
    assert model(x).shape == torch.Size([BATCH_SIZE, 1000])
    print(model(x).shape)
