import cv2
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

class getcolor(self):

    def __init__(self):
        return self

    def find_histogram(self,clt):
        """

        create a histogram with k clusters

        :param: clt

        :return:hist

        """

        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        hist = hist.astype("float")

        hist /= hist.sum()

        return hist


    def plot_colors2(self,hist, centroids):

        bar = np.zeros((50, 300, 3), dtype="uint8")

        startX = 0

        for (percent, color) in zip(hist, centroids):

            # plot the relative percentage of each cluster

            endX = startX + (percent * 300)

            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),

                        color.astype("uint8").tolist(), -1)

            startX = endX

        # return the bar chart

        return bar


    def gcolor(self,left,right,top,bottom,path):
        
        image = cv2.imread(path)

        
        middleh = int((bottom-top)/2 + top)
        middlew = int((right-left)/2 + left)
        ntop = middleh - int((middleh-top)*(2/10))
        nbottom = int((bottom-middleh)*(2/10)) + (middleh)
        nleft = middlew - int((middlew-left)*(2/10))
        nright = int((right-middlew)*(2/10)) + (middlew)
        croppedimage = image[ntop:nbottom, nleft:nright]
        # cv2.imshow('image', croppedimage)
        # cv2.waitKey(0)
        img = cv2.cvtColor(croppedimage, cv2.COLOR_BGR2RGB)


        # represent as row*column,channel number
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        clt = KMeans(n_clusters=3)  # cluster number

        clt.fit(img)

        print(clt.cluster_centers_)
        d = {}
        with open("colors.json", 'r') as fp:
            d = json.load(fp)

        y = []
        for k in d.keys():
            # print(k)
            i = int(int(k)/(1000*1000))
            j = int(int(k)/1000) - i*1000
            l = int(int(k)) - i*1000*1000 - j*1000
            y.append(np.array([i, j, l]))

        hist = find_histogram(clt)

        x=np.argmax(hist)

        maxcolor = np.array(clt.cluster_centers_[x])

        cmax = 0
        m = 0
        for i in y:
            c = np.dot(i, maxcolor)/((np.linalg.norm(i))*(np.linalg.norm(maxcolor)))
            # c = (i[0]*maxcolor[0] + i[1]*maxcolor[1] + i[2]*maxcolor[2])/((np.linalg.norm(i))*(np.linalg.norm(maxcolor)))
            # print(c)
            if(c > cmax):
                cmax = c
                m = i
        m = m[0]*1000*1000 + m[1]*1000 + m[2]
        print(d[str(m)])


    # print(hist)

    # bar = plot_colors2(hist, clt.cluster_centers_)


    # plt.axis("off")

    # plt.imshow(bar)

    # plt.show()
