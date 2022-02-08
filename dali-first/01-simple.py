from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "images"
max_batch_size = 8


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels


pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
pipe.build()

pipe_out = pipe.run()
print(pipe_out)

images, labels = pipe_out
print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))

import numpy as np

labels_tensor = labels.as_tensor()

print (labels_tensor.shape())
print (np.array(labels_tensor))
