# error 발생
import torch
from effdet_model_1 import EfficientDetModel

model = EfficientDetModel(
    num_classes=1,
    img_size=512
    )

model.load_state_dict(torch.load('trained_effdet'))
model.eval()

scripted = torch.jit.script(model)
scripted.save('scripted.torchscript')
