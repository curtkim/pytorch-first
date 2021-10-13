import torch
from pytorch_lightning import Trainer


from dataset_adaptor import CarsDatasetAdaptor
from effdet_datamodule import EfficientDetDataModule
from effdet_model_1 import EfficientDetModel

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import random

# import warnings
# warnings.filterwarnings("ignore")
# warnings.filterwarnings(
#     "ignore",
#     "Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesirable behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.",
#     UserWarning,
#     "setuptools.distutils_patch"
# )


def split_dataframe(df, train_size):
    images = df.image.unique().tolist()
    random.shuffle(images)

    train_count = int(len(images) * train_size)
    train_list = images[:train_count]
    condition = df['image'].isin(train_list)
    return df.loc[condition], df.loc[~condition]


def train(resume_model_path=None):
    dataset_path = Path('/data/datasets/car_object_detection')
    list(dataset_path.iterdir())

    df = pd.read_csv(dataset_path/'train_solution_bounding_boxes (1).csv')
    print(len(df))
    #train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    #print(len(df), len(train_df), len(test_df))

    train_data_path = dataset_path/'training_images'
    test_data_path = dataset_path/'testing_images'

    train_df, valid_df = split_dataframe(df, 0.8)
    print('train:valid', len(train_df), len(valid_df))

    cars_train_ds = CarsDatasetAdaptor(train_data_path, train_df)
    cars_valid_ds = CarsDatasetAdaptor(train_data_path, valid_df)
    print('train:valid', len(cars_train_ds), len(cars_valid_ds))

    dm = EfficientDetDataModule(train_dataset_adaptor=cars_train_ds,
            validation_dataset_adaptor=cars_valid_ds,
            num_workers=4,
            batch_size=2)
    model = EfficientDetModel(
        num_classes=1,
        img_size=512
        )

    trainer = Trainer(
        #gpus=[0, 1],
        gpus=None,
        max_epochs=20,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, dm)
    torch.save(model.state_dict(), 'trained_effdet')


if __name__ == '__main__':
    train()
    #freeze_support()

