from pathlib import Path
import pandas as pd
from dataset_adaptor import CarsDatasetAdaptor
from effdet_transformations import get_valid_transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



dataset_path = Path('/data/datasets/car_object_detection')
list(dataset_path.iterdir())

df = pd.read_csv(dataset_path/'train_solution_bounding_boxes (1).csv')

train_data_path = dataset_path/'training_images'
cars_train_ds = CarsDatasetAdaptor(train_data_path, df)

image1, truth_bboxes1, _, _ = cars_train_ds.get_image_and_labels_by_idx(0)
image2, truth_bboxes2, _, _ = cars_train_ds.get_image_and_labels_by_idx(1)
images = [image1, image2]

print(image1.width, image1.height)

resize = A.Resize(height=512, width=512, p=1)
resized = resize.apply(img=image1)
print(resized.width, resized.height)

to_tensor = ToTensorV2(p=1)


