import cv2
import numpy as np
import os
import pandas as pd
import csv

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

img_path = 'input/images/'
train = pd.read_csv('input/train.csv')
species = train.species.sort_values().unique()
sift = cv2.ORB_create()

dico = []

for leaf in train.id:
    print ("leaf",leaf)
    img = cv2.imread(img_path + str(leaf) + ".jpg")
    kp, des = sift.detectAndCompute(img, None)

    for d in des:
        dico.append(d)

k = np.size(species) * 10

batch_size = np.size(os.listdir(img_path)) * 3
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)

kmeans.verbose = False

histo_list = []

train = train.as_matrix()

for item in train:
    leaf = item[0]
    print ('leaf', leaf)
    species = item[1]
    img = cv2.imread(img_path + str(leaf) + ".jpg")
    kp, des = sift.detectAndCompute(img, None)

    histo = np.zeros(k+2)
    nkp = np.size(kp)

    for d in des:
        idx = kmeans.predict([d])
        histo[idx] += float(1)/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

    histo = histo.tolist()
    histo = [leaf, species] + histo

    histo_list.append(histo)

X = np.array(histo_list)
Y = []

X = pd.DataFrame(X)
X.to_csv("train_visord-orb.csv")
