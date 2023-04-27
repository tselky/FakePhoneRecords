# import module
import random as r
from numpy.random import choice

local_codes = ['231', '248', '269', '313', '517', '586', '616', '734', '810', '906', '947', '989']


sampleList = [str(r.sample(range(1000000000, 9999999999), 1)), str(r.choice(local_codes))+str(r.sample(range(1000000, 9999999), 1)[0])]
randomNumberList = choice(
    sampleList, 1, p=[0.2, 0.8])

print(randomNumberList)

s1 = str(2486755511)

result = s1.startswith(tuple(local_codes))

# If result is true print Yes
if result:
    print("Yes")
else:
    print("No")




