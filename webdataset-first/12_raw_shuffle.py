import webdataset as wds

my_iter = wds.shuffle(iter([1, 2, 3, 4, 5]))

for item in my_iter:
    print(item)

# 1 3 5 2 4
