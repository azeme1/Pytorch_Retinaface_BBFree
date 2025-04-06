import torch


class BasicBlock_X1(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=(
            3, 3), padding=(
            1, 1), padding_mode='reflect'):
        super(BasicBlock_X1, self).__init__()
        self.conv_00 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_00 = torch.nn.BatchNorm2d(out_channels)
        self.dropout2d = torch.nn.Dropout2d(p=0.05)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.bn_00(self.conv_00(x)))
        x = self.dropout2d(x)
        return x


class DownBlock_X1(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=(
            3, 3), padding=(
            1, 1), padding_mode='reflect'):
        super(DownBlock_X1, self).__init__()
        self.conv_00 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_00 = torch.nn.BatchNorm2d(out_channels, momentum=0.01)
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.dropout2d = torch.nn.Dropout2d(p=0.05)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.bn_00(self.conv_00(x)))
        x = self.dropout2d(x)
        y = x
        x = self.maxpool(x)
        return x, y


class UpBlock_X1(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=(
            3, 3), padding=(
            1, 1), padding_mode='reflect'):
        super(UpBlock_X1, self).__init__()
        self.conv_00 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_00 = torch.nn.BatchNorm2d(out_channels, momentum=0.01)
        self.upsample = torch.nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.dropout2d = torch.nn.Dropout2d(p=0.05)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.bn_00(self.conv_00(x)))
        x = self.dropout2d(x)
        x = self.upsample(x)
        return x


class BasicBlock_X2(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=(
            3, 3), padding=(
            1, 1), padding_mode='reflect'):
        super(BasicBlock_X2, self).__init__()
        self.conv_00 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_00 = torch.nn.BatchNorm2d(out_channels)
        self.conv_01 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_01 = torch.nn.BatchNorm2d(out_channels)
        self.dropout2d = torch.nn.Dropout2d(p=0.05)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.bn_00(self.conv_00(x)))
        x = x + self.activation(self.bn_01(self.conv_01(x)))
        x = self.dropout2d(x)
        return x


class DownBlock_X2(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=(
            3, 3), padding=(
            1, 1), padding_mode='reflect'):
        super(DownBlock_X2, self).__init__()
        self.conv_00 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_00 = torch.nn.BatchNorm2d(out_channels, momentum=0.01)
        self.conv_01 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_01 = torch.nn.BatchNorm2d(out_channels, momentum=0.01)
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.dropout2d = torch.nn.Dropout2d(p=0.05)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.bn_00(self.conv_00(x)))
        x = x + self.activation(self.bn_01(self.conv_01(x)))
        x = self.dropout2d(x)
        y = x
        x = self.maxpool(x)
        return x, y


class UpBlock_X2(torch.torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=(
            3, 3), padding=(
            1, 1), padding_mode='reflect'):
        super(UpBlock_X2, self).__init__()
        self.conv_00 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_00 = torch.nn.BatchNorm2d(out_channels, momentum=0.01)
        self.conv_01 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel,
            padding=padding,
            padding_mode=padding_mode)
        self.bn_01 = torch.nn.BatchNorm2d(out_channels, momentum=0.01)
        self.upsample = torch.nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.dropout2d = torch.nn.Dropout2d(p=0.05)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.bn_00(self.conv_00(x)))
        x = x + self.activation(self.bn_01(self.conv_01(x)))
        x = self.dropout2d(x)
        x = self.upsample(x)
        return x


class UNet_X1(torch.torch.nn.Module):
    def __init__(
            self,
            in_channel,
            out_channes,
            output_activation=None,
            use_batch_norm=False):
        super(UNet_X1, self).__init__()

        self.use_batch_norm = use_batch_norm

        DownBlock, BasicBlock, UpBlock = self.get_basic_blocks()
        d_list, u_list = self.get_basic_weights()
        dc_00, dc_01, dc_02, dc_03, dc_04, dc_05 = d_list
        uc_05, uc_04, uc_03, uc_02, uc_01, uc_00 = u_list

        self.d_00 = DownBlock(in_channel, dc_00)
        self.d_01 = DownBlock(dc_00, dc_01)
        self.d_02 = DownBlock(dc_01, dc_02)
        self.d_03 = DownBlock(dc_02, dc_03)
        self.d_04 = DownBlock(dc_03, dc_04)
        self.d_05 = DownBlock(dc_04, dc_05)

        self.up_05 = UpBlock(dc_05, uc_05)
        self.up_04 = UpBlock(dc_05 + uc_05, uc_04)
        self.up_03 = UpBlock(dc_04 + uc_04, uc_03)
        self.up_02 = UpBlock(dc_03 + uc_03, uc_02)
        self.up_01 = UpBlock(dc_02 + uc_02, uc_01)
        self.up_00 = UpBlock(dc_01 + uc_01, uc_00)

        self.bn_first = torch.torch.nn.BatchNorm2d(in_channel, momentum=0.01)
        self.last_block = BasicBlock(uc_00 + dc_00, uc_00)
        self.conv_last = torch.torch.nn.Conv2d(uc_00, out_channes, (3, 3), padding=1)
        self.output_activation = output_activation

        self.conv_c_00 = torch.torch.nn.Conv2d(dc_05, dc_05 // 8, (8, 8), padding=0)
        self.conv_c_01 = torch.torch.nn.Conv2d(dc_05 // 8, 1, (1, 1), padding=0)

        self.conv_d_00 = torch.torch.nn.Conv2d(dc_05, dc_05 // 2, (8, 8), padding=0)
        self.conv_d_01 = torch.torch.nn.Conv2d(dc_05 // 2, dc_05 // 4, (1, 1), padding=0)
        self.conv_d_02 = torch.torch.nn.Conv2d(dc_05 // 4, dc_05 // 8, (1, 1), padding=0)
        self.conv_d_03 = torch.torch.nn.Conv2d(dc_05 // 8, 4, (1, 1), padding=0)

    def get_basic_blocks(self):
        return DownBlock_X2, BasicBlock_X2, UpBlock_X2

    def get_basic_weights(self):
        d_list = [16, 24, 32, 48, 64, 96]
        u_list = [96, 64, 48, 32, 24, 16]
        return d_list, u_list

    def forward(self, x):

        if self.use_batch_norm:
            x = self.bn_first(x)
        x, x_00 = self.d_00(x)
        x, x_01 = self.d_01(x)
        x, x_02 = self.d_02(x)
        x, x_03 = self.d_03(x)
        x, x_04 = self.d_04(x)
        x, x_05 = self.d_05(x)  # 8x8

        x = self.up_05(x)
        x = torch.cat([x, x_05], dim=1)
        x = self.up_04(x)
        x = torch.cat([x, x_04], dim=1)
        x = self.up_03(x)
        x = torch.cat([x, x_03], dim=1)
        x = self.up_02(x)
        x = torch.cat([x, x_02], dim=1)
        x = self.up_01(x)
        x = torch.cat([x, x_01], dim=1)
        x = self.up_00(x)
        x = torch.cat([x, x_00], dim=1)

        x = self.last_block(x)
        x = self.conv_last(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
