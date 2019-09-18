import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

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

def get_label():
    return np.loadtxt("../data/masquerade/label.txt")

def main():
    cmd_list,dist = convert_to_cmdlist_new("../data/masquerade/User3")
    X = get_feature_new(cmd_list,dist)
    Y = get_label()
    Y = np.vstack((np.zeros((50, 1)), Y[:, 2].reshape(Y.shape[0], 1))).ravel()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4)

    clf = GaussianNB().fit(X_train,Y_train)   ###高斯贝叶斯分类器

    Y_pred = clf.predict(X_test)
    score = np.mean(Y_pred==Y_test)
    print("score=",score)
main()
