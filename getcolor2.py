import json
from collections import Counter
import operator
from PIL import Image
import cv2
x = {}
with open("result0.json") as fp:
    x = json.load(fp)
image = Image.open("x.jpg")

top = x["Top"]
bottom = x["Bottom"]
right = x["Right"]
left = x["Left"]

print(top,bottom,right,left)
middleh = int((bottom-top)/2 + top)
middlew = int((right-left)/2 + left)
print(middleh,middlew)
ntop = middleh - int((middleh-top)*(1/10)) 
nbottom = int((bottom-middleh)*(1/10)) + (middleh)
nleft = middlew - int((middlew-left)*(1/10)) 
nright = int((right-middlew)*(1/10)) + (middlew)
cropped_img = image.crop((nleft,ntop,nright,nbottom))
cropped_img.show()
print(ntop,nbottom,nleft,nright)
lines3 = []
wcounter = 0
tcounter = 0
for i in range(int(ntop),int(nbottom)):
    for j in range(int(nleft),int(nright)):
        l = list(image.getpixel((j,i)))
        l[0] = int(l[0])
        l[1] = int(l[1])
        l[2] = int(l[2])
        a = l[0]*1000*1000 + l[1]*1000 + l[2]
        if(a > 210210210):
            wcounter = wcounter + 1
        tcounter = tcounter + 1
        lines3.append(a)
ctr = Counter(lines3)
with open("counter.json", 'w') as fp:
    json.dump(ctr,fp,indent=4)
ctr1 = sorted(ctr.items(),key=operator.itemgetter(1))
with open("counter1.json","w") as fp:
    json.dump(ctr1,fp,indent=4)

print(wcounter/tcounter)