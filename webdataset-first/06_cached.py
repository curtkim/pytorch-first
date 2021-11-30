import webdataset as wds


# just using one URL for demonstration
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
dataset = wds.WebDataset(url, cache_dir="./cache")

print("=== first pass")

for sample in dataset:
    pass

print("=== second pass")

for i, sample in enumerate(dataset):
    for key, value in sample.items():
        print(key, repr(value)[:50])
    print()
    if i >= 3: break