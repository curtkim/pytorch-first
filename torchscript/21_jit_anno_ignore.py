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


def test_scripted_module():
    m = torch.jit.script(MyModule())

    # This is OK since `training_method` is not saved, the call is replaced
    # with a `raise`.
    m.save("m.pt")

    sample = torch.arange(5)
    try:
        print(m(sample))
    except torch.jit.Error as err:
        print('catched')


def test_raw_module():
    sample = torch.arange(5)

    mym = MyModule()
    assert mym.training
    mym.eval()
    assert not mym.training
    print(mym(sample))
    mym.train()
    assert mym.training


test_raw_module()
test_scripted_module()
