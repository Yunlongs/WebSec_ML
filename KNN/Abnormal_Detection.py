from nltk.probability import FreqDist
import numpy as  np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score


def convert_to_cmdlist(filename):
    cmd_list =[]
    dist = []
    with open(filename) as f:
        x = []
        for i,line in enumerate(f.readlines()):
            x.append(line.strip())
            dist.append(line.strip())
            if i%100 == 0:
                cmd_list.append(x)
                x = []
    fdist = list(FreqDist(dist).keys())
    dist_max50 = set(fdist[:50])  ### 最频繁的50个命令
    dist_min50 = set(fdist[-50:]) ### 最不频繁的50个命令
    return cmd_list,dist_max50,dist_min50


def get_feature(cmd_list,dist_max50,dist_min50):
    x = []
    for cmd in cmd_list:
        f1 = len(set(cmd))              ### 特征（1）
        f2 = list(FreqDist(cmd).keys())[:10]
        f3 = list(FreqDist(cmd).keys())[-10:]
        f2 = len(set(f2) & dist_max50)  ###特征（2）
        f3 = len(set(f3) & dist_min50)  ###特征（3）
        f = [f1,f2,f3]
        x.append(f)
    return x

def get_label():
    return np.loadtxt("../data/masquerade/label.txt")

def KNN_train_1():
    N = 90
    cmd_list,dist_max50,dist_min50 = convert_to_cmdlist("../data/masquerade/User3")
    X = get_feature(cmd_list,dist_max50,dist_min50)
    Y = get_label()
    Y = np.vstack((np.zeros((50,1)),Y[:,2].reshape(Y.shape[0],1)))
    X_train = X[:N]
    X_test = X[N:]
    Y_train = Y[:N].ravel()
    Y_test = Y[N:].ravel()

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,Y_train)
    Y_pred = neigh.predict(X_test)

    score = np.mean(Y_test==Y_pred) *100
    print("score = ",score)
#KNN_train_1()


def convert_to_cmdlist_new(filename):
    cmd_list = []
    dist = []
    with open(filename) as f:
        x = []
        for i, line in enumerate(f.readlines()):
            x.append(line.strip())
            dist.append(line.strip())
            if i % 100 == 0:
                cmd_list.append(x)
                x = []
    return cmd_list,list(set(dist))

def get_feature_new(cmd_list,dist):
    matrix = np.zeros((len(cmd_list),len(dist)))
    for i,cmd in enumerate(cmd_list):
        for j,x in enumerate(dist):
            if x in cmd:
                matrix[i,j] += 1
    return matrix

def KNN_train_2():
    N=90
    cmd_list,dist = convert_to_cmdlist_new("../data/masquerade/User3")
    X = get_feature_new(cmd_list,dist)
    Y = get_label()
    Y = np.vstack((np.zeros((50, 1)), Y[:, 2].reshape(Y.shape[0], 1)))

    neigh = KNeighborsClassifier(n_neighbors=3)
    score = cross_val_score(neigh,X,Y.ravel(),cv=10,n_jobs=1)
    print(score)
KNN_train_2()