# error 발생
import torch
from effdet import DetBenchTrain, EfficientDet

from effdet_create_model import create_model

"""
from effdet_model_1 import EfficientDetModel

model = EfficientDetModel(
    num_classes=1,
    img_size=512
    )

model.load_state_dict(torch.load('trained_effdet'))
model.eval()
"""
model = create_model()

#model2: DetBenchTrain = model.model
model3: EfficientDet = model.model
efficient_det_backbone = model.model.backbone     # 성공
efficient_det_fpn = model.model.fpn               # 성공
efficient_det_class_net = model.model.class_net   # 실패
efficient_det_box_net = model.model.box_net       # 실패

print(efficient_det_box_net.bn_level_first)
scripted = torch.jit.script(efficient_det_class_net)


# Error of model3
# ======
# Could not export Python function call '_forward'. Remove calls to Python functions before export.
# Did you forget to add @script or @script_method annotation?
# If this is a nn.ModuleList, add it to __constants__:
#   File "/home/curt/.pyenv/versions/3.8.9/lib/python3.8/site-packages/effdet/efficientdet.py", line 454
#
#   HeadNet
#             return self._forward_level_first(x)
#         else:
#             return self._forward(x)
#                    ~~~~~~~~~~~~~ <--- HERE

scripted.save('scripted.torchscript')
