import torchvision
from torchinfo import summary

model = torchvision.models.resnet152()
summary(model, (1, 3, 224, 224), depth=3)


# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# ResNet                                   --                        --
# ├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
# ├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
# ...
# │    │    └─ReLU: 3-454                  [1, 2048, 7, 7]           --
# ├─AdaptiveAvgPool2d: 1-9                 [1, 2048, 1, 1]           --
# ├─Linear: 1-10                           [1, 1000]                 2,049,000
# ==========================================================================================
# Total params: 60,192,808
# Trainable params: 60,192,808
# Non-trainable params: 0
# Total mult-adds (G): 11.51
# ==========================================================================================
# Input size (MB): 0.60
# Forward/backward pass size (MB): 360.87
# Params size (MB): 240.77
# Estimated Total Size (MB): 602.25
# ==========================================================================================
#
# Process finished with exit code 0
