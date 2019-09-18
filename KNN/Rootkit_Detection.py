import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def get_data():
    x = []
    Y = []
    with open("../data/kddcup.data_10_percent_corrected") as f:
        for line in f.readlines():
            line = line.strip().split(',')
            if line[2] == 'telnet' and line[41] in['rootkit.','normal.']:
                x.append(line)
                if line[41] == 'rootkit.':
                    Y.append(1)
                else:
                    Y.append(0)
    return x,Y

def get_feature(x):
    X = np.array(x)
    X = X[:,9:21]
    return X

def KNN_train_1():
    x,Y = get_data()
    X = get_feature(x)

    neigh = KNeighborsClassifier(n_neighbors=3)

    score = cross_val_score(neigh,X,Y,cv=10)
    print(score)
KNN_train_1()



