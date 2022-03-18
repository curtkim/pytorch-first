from torchdata.datapipes.iter import IterableWrapper, Mapper

dp = IterableWrapper(range(10))
map_dp = dp.map(lambda x: x+1)

print(list(map_dp))

filter_dp = map_dp.filter(lambda x: x%2 == 0)
print(list(filter_dp))

