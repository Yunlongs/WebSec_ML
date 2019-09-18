import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1500
x,y = make_blobs(n_samples,random_state=170)

y_pred = KMeans(n_clusters=3,random_state=170).fit_predict(x)

plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.title("hello world!")
plt.show()