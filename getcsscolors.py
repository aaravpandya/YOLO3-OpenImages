import urllib3 as urllib
from bs4 import BeautifulSoup
import json

def hex_to_rgb(hex):
    h = hex.lstrip('#')
    rgb = list(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
    rgbn = rgb[0]*1000*1000 + rgb[1]*1000 + rgb[2]
    return rgbn
http = urllib.PoolManager()
r = http.request('GET', 'https://www.w3schools.com/cssref/css_colors.asp')
soup = BeautifulSoup(r.data,features="html5lib")
d = {}
l = []
for link in soup.findAll('a'):
    try:
        if(link['target']=="_blank"):
            l.append(link.text)
    except:
        continue
l = l[:len(l)-3]
print(l)
d = {}
for i in range(0,len(l)):
    if(l[i][0] == "#"):
        continue
    print(l[i+1])
    h = hex_to_rgb(l[i+1])
    d[h] = l[i]

with open("colors.json",'w') as fp:
    json.dump(d,fp,indent=4)



