# error 발생
import torch

from src.effdet_model_1 import EfficientDetModel

model = EfficientDetModel(
    num_classes=1,
    img_size=512
    )

model.load_state_dict(torch.load('../trained_effdet'))
model.eval()
print(type(model))
print(type(model.model))
print(type(model.model.model))
target_model = model.model.model

scripted = torch.jit.script(target_model)
scripted.save('../scripted.torchscript')

