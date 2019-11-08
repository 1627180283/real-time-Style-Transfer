from torch import nn


# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding=kernel_size//2)

    def forward(self, x):
        out = self.conv2d(x)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, upsample=None, size=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.size = size
        if upsample and size is None:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        if upsample is None and size:
            self.upsample = nn.Upsample(size=size, mode="nearest")
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding=kernel_size//2)

    def forward(self, x):
        if self.upsample or self.size:
            x = self.upsample(x)
        out = self.conv2d(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))

        out = self.in2(self.conv2(out))

        out = out + residual
        out = self.relu(out)
        return out


# Image Transform Network
class ImageTransformNet(nn.Module):
    def __init__(self, input_size=(256, 256)):
        super(ImageTransformNet, self).__init__()

        self.size = input_size
        self.tmp_size = (input_size[0] // 2, input_size[1] // 2)

        # nonlineraity
        self.relu = nn.ReLU()

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, size=self.tmp_size)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, size=self.size)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        y = self.deconv1(y)

        return y
