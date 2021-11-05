# error 발생
import torch
from effdet import DetBenchTrain, EfficientDet

from effdet_create_model import create_model
from effdet_model_1 import EfficientDetModel


model = EfficientDetModel(
    num_classes=1,
    img_size=512
    )

model.load_state_dict(torch.load('trained_effdet'))
model.eval()
print(type(model))
print(type(model.model))
print(type(model.model.model))
target_model = model.model.model

traced = torch.jit.trace(target_model, torch.rand(2, 3, 512, 512))
traced.save('traced.torchscript')

"""
아래 코드에서 warnning이 발생
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


에러메시지
/home/curt/.pyenv/versions/3.8.7/lib/python3.8/site-packages/timm/models/layers/padding.py:19: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)
/home/curt/.pyenv/versions/3.8.7/lib/python3.8/site-packages/timm/models/layers/padding.py:19: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)
/home/curt/.pyenv/versions/3.8.7/lib/python3.8/site-packages/timm/models/layers/padding.py:31: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if pad_h > 0 or pad_w > 0:
/home/curt/.pyenv/versions/3.8.7/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /pytorch/aten/src/ATen/native/BinaryOps.cpp:467.)
  return torch.floor_divide(self, other)
/home/curt/.pyenv/versions/3.8.7/lib/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
/home/curt/.pyenv/versions/3.8.7/lib/python3.8/site-packages/torch/jit/_trace.py:952: TracerWarning: Encountering a list at the output of the tracer might cause the trace to be incorrect, this is only valid if the container structure does not change based on the module's inputs. Consider using a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead). If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior.
  module._c._create_method_from_trace(
"""