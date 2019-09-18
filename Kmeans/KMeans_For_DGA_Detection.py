import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_dga(file_crypto,file_post):
    x1 = []
    x2 = []
    with open(file_crypto,'r') as f:
        for line in f.readlines():
            x1.append(line.strip().split(',')[0])
    with open(file_post,'r') as f:
        for line in f.readlines():
            x2.append(line.strip().split(',')[0])
    return x1,x2

def load_alexa(filename):
    x3 =[]
    with open(filename,'r') as f:
        for line in f.readlines():
            x3.append(line.strip().split(',')[1])
    return x3

def get_feature():
    x1,x2 = load_dga("../data/dga/dga-cryptolocke-50.txt","../data/dga/dga-post-tovar-goz-50.txt")
    x3 = load_alexa("../data/dga/top-100.csv")

    y1 = [2]*len(x1)
    y2 = [3]*len(x2)
    y3 = [0] * len(x3)

    x = x1+x2+x3
    y = y1+y2+y3

    cv = CountVectorizer(decode_error='ignore',ngram_range=(2,2),token_pattern="\w")
    X = cv.fit_transform(x).toarray()
    return X,y

X,Y = get_feature()

model = KMeans(n_clusters=2)
Y_pred = model.fit_predict(X)

tsne = TSNE(learning_rate=100)
x = tsne.fit_transform(X)

for i,label in enumerate(x):
    x1,x2 = x[i]
    if Y_pred[i] == 1:
        plt.scatter(x1,x2,marker='o',color = 'r')
    else:
        plt.scatter(x1,x2,marker='x',color='b')
plt.show()