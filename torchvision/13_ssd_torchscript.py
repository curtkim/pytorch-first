import torchvision
import torch


def param_count(model):
    return sum(p.numel() for p in model.parameters())


model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

print("backbone count", param_count(model.backbone))
print("head count", param_count(model.head))
print("anchor count", param_count(model.anchor_generator))
# backbone count 22,943,936
# head count 12,697,890
# anchor count 0



img = torch.rand(1, 3, 2139, 3500)
print(img.shape)


# scripted_model = torch.jit.script(model)
# #print(scripted_model.code)
# scripted_model.save('ssd300_vgg16.torchscript')
# # 137M
#
# scripted_backbone = torch.jit.script(model.backbone)
# scripted_backbone.save('ssd300_vgg16_backbone.torchscript')
# # 88M
#
# scripted_head = torch.jit.script(model.head)
# scripted_head.save('ssd300_vgg16_head.torchscript')
# # 49M


ssdlite = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
print("ssdlite count", param_count(ssdlite))
scripted_ssdlite = torch.jit.script(ssdlite)
scripted_ssdlite.save('ssdlite.torchscript')
