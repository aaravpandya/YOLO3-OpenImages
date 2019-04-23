import json
import numpy as np
d = {}
with open("colors.json",'r') as fp:
    d = json.load(fp)

y = []
for k in d.keys():
    # print(k)
    i = int(int(k)/(1000*1000))
    j = int(int(k)/1000) - i*1000
    l = int(int(k)) - i*1000*1000 - j*1000
    y.append(np.array([i,j,l]))