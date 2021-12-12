from pathlib import Path
import pandas as pd

import torch
from src.dataset_adaptor import CarsDatasetAdaptor
from predictor import predict
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config
from effdet.anchors import Anchors, AnchorLabeler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


# %%
num_classes = 1
image_size = 512
architecture = 'tf_efficientnetv2_l'

# %%
dataset_path = Path('/data/datasets/car_object_detection')
list(dataset_path.iterdir())

df = pd.read_csv(dataset_path/'train_solution_bounding_boxes (1).csv')

train_data_path = dataset_path/'training_images'
cars_train_ds = CarsDatasetAdaptor(train_data_path, df)

# %%
image1, truth_bboxes1, _, _ = cars_train_ds.get_image_and_labels_by_idx(0)
image2, truth_bboxes2, _, _ = cars_train_ds.get_image_and_labels_by_idx(1)

print(type(image1))

# %%

transform = A.Compose([
    A.Resize(height=image_size, width=image_size, p=1),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2(p=1),
])


transformed1 = transform(image=np.array(image1))
transformed1_image = transformed1["image"]
transformed2 = transform(image=np.array(image2))
transformed2_image = transformed2["image"]

images = torch.stack([transformed1_image, transformed2_image])
print(images.shape)

# %%
scripted_efficient_det = torch.jit.load('scripted.torchscript')
scripted_efficient_det.eval()

# %%
efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(
    name='tf_efficientnetv2_l',
    backbone_name='tf_efficientnetv2_l',
    backbone_args=dict(drop_path_rate=0.2),
    num_classes=num_classes,
    url='', )

config = get_efficientdet_config(architecture)
config.update({'num_classes': num_classes})
config.update({'image_size': (image_size, image_size)})
# torchscript때문에 추가됨.
config.update({'head_bn_level_first': True})

#%%
anchors = Anchors.from_config(config)
anchor_labeler = AnchorLabeler(anchors, num_classes, match_threshold=0.5)

#%%
image_sizes = [(image1.size[1], image1.size[0]), (image2.size[1], image2.size[0])]
scaled_bboxes, predicted_class_labels, predicted_class_confidences = predict(images, image_sizes, scripted_efficient_det, anchors.boxes, config)
print(scaled_bboxes)
print(predicted_class_labels)
print(predicted_class_confidences)
