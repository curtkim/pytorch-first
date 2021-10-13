from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from time import perf_counter


def timer(f, *args):
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))


script_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', torchscript=True)
script_model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = script_tokenizer.tokenize(text)

# Masking one of the input tokens
masked_index = 8

tokenized_text[masked_index] = '[MASK]'

indexed_tokens = script_tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# Example 1.1 BERT on CPU
native_model = BertModel.from_pretrained("bert-base-uncased")
print(np.mean([timer(native_model, tokens_tensor, segments_tensors) for _ in range(100)]))


# Example 1.2 BERT on GPU
# Both sample data model need be on the GPU device for the inference to take place
native_gpu = native_model.cuda()
tokens_tensor_gpu = tokens_tensor.cuda()
segments_tensors_gpu = segments_tensors.cuda()
print(np.mean([timer(native_gpu, tokens_tensor_gpu, segments_tensors_gpu) for _ in range(100)]))


# Example 2.1 torch.jit.trace on CPU
traced_model = torch.jit.trace(script_model, [tokens_tensor, segments_tensors])
print(np.mean([timer(traced_model, tokens_tensor, segments_tensors) for _ in range(100)]))


# Example 2.2 torch.jit.trace on GPU
traced_model_gpu = torch.jit.trace(script_model.cuda(), [tokens_tensor.cuda(), segments_tensors.cuda()])
print(np.mean([timer(traced_model_gpu, tokens_tensor.cuda(), segments_tensors.cuda()) for _ in range(100)]))

# 92.096
# 16.347
# 77.887
# 54.216        why so slow