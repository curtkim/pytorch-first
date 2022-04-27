import torch


class Add(torch.nn.Module):
    def forward(self, x):
        return x+1


inputs = (torch.arange(4, dtype=torch.float32))
print(Add()(inputs))
torch.onnx.export(Add(), inputs, '01_add.onnx', opset_version=11)
