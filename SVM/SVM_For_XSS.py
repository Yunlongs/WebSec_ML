import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from  sklearn.svm import SVC


def get_len(url):
    return len(url)

def get_url_count(url):
    pattern = re.compile('(https://)|(http://)',re.IGNORECASE)
    if re.search(pattern,url):
        return 1
    else:
        return 0

def get_evil_char(url):
    pattern = re.compile("[<>,\'\"/]",re.IGNORECASE)
    return len(re.findall(pattern,url))

def get_evil_word(url):
    pattern = re.compile("(alert)|(script=)|(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)",re.IGNORECASE)
    return len(re.findall(pattern,url))

def load_data(filename):
    x = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            x.append(line.strip())
    return x

def get_feature():
    x1 = load_data("../data/web-attack/xss-200000.txt")
    x2 = load_data("../data/web-attack/normal-10000.txt")
    x = x1+x2
    X = np.zeros((len(x),4))
    for i,url in enumerate(x):
        X[i,0] = get_len(url)
        X[i,1] = get_url_count(url)
        X[i,2] = get_evil_char(url)
        X [i,3] = get_evil_word(url)

    ### 标准化数据特征
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    Y = [1]*len(x1)+[0]*len(x2)

    return X,Y

def main():
    X,Y = get_feature()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

    clf = SVC(kernel='linear')
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    score = np.mean(Y_pred == Y_test)*100
    print("score=",score)
main()

