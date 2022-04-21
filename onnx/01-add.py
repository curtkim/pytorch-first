import torch


class Add(torch.nn.Module):
    def forward(self, x):
        return x+1

print(Add()(torch.arange(4)))


inputs = (torch.arange(4))
torch.onnx.export(Add(), inputs, '01_add.onnx', opset_version=11)
