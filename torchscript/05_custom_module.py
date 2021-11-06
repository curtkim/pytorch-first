# https://spell.ml/blog/pytorch-jit-YBmYuBEAACgAiv71

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit


class Conv2d(nn.Module):
    def __init__(
            self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.n_channels = n_channels
        self.weights = nn.Parameter(
            torch.Tensor(self.out_channels, self.n_channels, self.kernel_size ** 2)
        )

    def __repr__(self):
        return (
            f"Conv2d(n_channels={self.n_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size})"
        )

    def forward(self, x):
        width = self.calculate_new_width(x)
        height = self.calculate_new_height(x)
        windows = self.calculate_windows(x)

        result = torch.zeros(
            [x.shape[0] * self.out_channels, width, height],
            dtype=torch.float32, device=x.device
        )

        # import pdb; pdb.set_trace()
        for channel in range(x.shape[1]):
            for i_conv_n in range(self.out_channels):
                # print(channel, i_conv_n)
                xx = torch.matmul(windows[channel], self.weights[i_conv_n][channel])
                xx = xx.view((-1, width, height))

                xx_stride = slice(i_conv_n * xx.shape[0], (i_conv_n + 1) * xx.shape[0])
                result[xx_stride] += xx

        result = result.view((x.shape[0], self.out_channels, width, height))
        return result

    def calculate_windows(self, x):
        windows = F.unfold(
            x,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            stride=(self.stride, self.stride)
        )

        windows = (windows
                   .transpose(1, 2)
                   .contiguous().view((-1, x.shape[1], int(self.kernel_size ** 2)))
                   .transpose(0, 1)
                   )
        return windows

    def calculate_new_width(self, x):
        return (
                       (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1

    def calculate_new_height(self, x):
        return (
                       (x.shape[3] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1


class Conv2d_script(jit.ScriptModule):
    def __init__(
            self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.n_channels = n_channels
        self.weights = nn.Parameter(
            torch.Tensor(self.out_channels, self.n_channels, self.kernel_size ** 2)
        )

    def __repr__(self):
        return (
            f"Conv2d(n_channels={self.n_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size})"
        )

    @jit.script_method
    def forward(self, x):
        width = self.calculate_new_width(x)
        height = self.calculate_new_height(x)
        windows = self.calculate_windows(x)

        result = torch.zeros(
            [x.shape[0] * self.out_channels, width, height],
            dtype=torch.float32, device=x.device
        )

        for channel in range(x.shape[1]):
            for i_conv_n in range(self.out_channels):
                xx = torch.matmul(windows[channel], self.weights[i_conv_n][channel])
                xx = xx.view((-1, width, height))

                xx_stride = slice(i_conv_n * xx.shape[0], (i_conv_n + 1) * xx.shape[0])
                result[xx_stride] += xx

        result = result.view((x.shape[0], self.out_channels, width, height))
        return result

    def calculate_windows(self, x):
        windows = F.unfold(
            x,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            stride=(self.stride, self.stride)
        )

        windows = (windows
                   .transpose(1, 2)
                   .contiguous().view((-1, x.shape[1], int(self.kernel_size ** 2)))
                   .transpose(0, 1)
                   )
        return windows

    def calculate_new_width(self, x):
        return (
                       (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1

    def calculate_new_height(self, x):
        return (
                       (x.shape[3] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1


def test1():
    x = torch.randint(0, 255, (1, 3, 512, 512), device='cuda') / 255

    conv = Conv2d(3, 16, 3)
    conv.cuda()

    out = conv(x)
    out.mean().backward()


def test2():
    x = torch.randint(0, 255, (1, 3, 512, 512), device='cuda') / 255

    conv = Conv2d_script(3, 16, 3)
    conv.cuda()

    out = conv(x)
    out.mean().backward()


if __name__ == "__main__":
    import timeit

    # warmup
    timeit.timeit(lambda: test1(), number=1)

    print("plain_python", timeit.timeit(lambda: test2(), number=10))
    print("script_method", timeit.timeit(lambda: test1(), number=10))
