class VAEResBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=5) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=5,
        )
        self.bn1 = torch.nn.BatchNorm2d(in_channels * 2)
        self.conv2 = torch.nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=5,
        )
        self.bn2 = torch.nn.BatchNorm2d(in_channels * 2)
        self.conv_skip = torch.nn.Conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=5,
        )
    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(x)))
        out += self.conv_skip(x)
        return out