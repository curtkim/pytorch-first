from common_transformer import ReverseDataset


dataset = ReverseDataset(num_categories=10, seq_len=16, size=1)
inp_data, labels = dataset[0]
print("Input data:", inp_data)
print("Labels:    ", labels)
