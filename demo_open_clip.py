import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = open_clip.tokenize(["a diagram", "a dog", "a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(image_features.shape)
print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

