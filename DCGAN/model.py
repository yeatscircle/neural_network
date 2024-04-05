import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # input: N x channels x 64 x 64
            nn.Conv2d(
                in_channels,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # output: N x 1 x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )


class Generator(nn.Module):
    def __init__(self, channel_noise, channel_img, feature_dim):
        super().__init__()
        # input:N x noise x 1 x 1
        self.gen = nn.Sequential(
            # Transpose是上采样 O = (I - 1) * S - 2P + K + output_padding
            self._block(channel_noise, feature_dim * 16, 4, 1, 0),  # img:4x4
            self._block(feature_dim * 16, feature_dim * 8, 4, 2, 1),  # 8x8
            self._block(feature_dim * 8, feature_dim * 4, 4, 2, 1),  # 16x16
            self._block(feature_dim * 4, feature_dim * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(
                feature_dim * 2, channel_img, 4, 2, 1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()


