import csv
from collections import Counter
lines = []
with open("out.csv") as f:
    rr = csv.reader(f)
    lines = [r for r in rr]
lines2 = [x for x in lines if x]
lines3 = []
for l in lines2:
    try:
        l[0] = int(l[0])
        l[1] = int(l[1])
        l[2] = int(l[2])
        a = l[0]*1000*1000 + l[1]*1000 + l[2]
        lines3.append(a)
    except:
        continue
print(lines2[0])
print(lines3[0])
ctr = Counter(lines3)
