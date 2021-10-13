import torch
import torch.nn as nn


class MyModule(nn.Module):
    @torch.jit.ignore(drop=True)
    def training_method(self, x):
        import pdb
        pdb.set_trace()

    def forward(self, x):
        if self.training:
            self.training_method(x)
        return x


m = torch.jit.script(MyModule())

# This is OK since `training_method` is not saved, the call is replaced
# with a `raise`.
m.save("m.pt")
