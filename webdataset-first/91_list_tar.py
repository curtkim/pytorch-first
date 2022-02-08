import json
import tarfile

tar = tarfile.open("openimages-train-000000.tar")

for info in tar.getmembers():
    print(info.name, info.type, info.size, 'bytes')
    if info.name.endswith('json'):
        f = tar.extractfile(info)
        print(json.load(f))

print('len', len(tar.getmembers()))
tar.close()
