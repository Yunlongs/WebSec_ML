import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def get_noranl_data(rootdir):
    file_list = os.listdir(rootdir)
    x = []
    y =[]
    for file in file_list:
        file_path = os.path.join(rootdir,file)
        if os.path.isfile(file_path):
            with open(file_path,'r') as f:
                x.append(f.read())
                y.append(0)
    return x,y

def dirlist(rootdir):
    all_file = []
    path_list = os.listdir(rootdir)
    for path in path_list:
        file_path = os.path.join(rootdir,path)
        if os.path.isdir(file_path):
            file_path += '/'
            all_file += dirlist(file_path)
        else:
            all_file.append(file_path)
    return all_file

def get_attack_data(rootdir):
    all_file = dirlist(rootdir)
    x = []
    y =[]
    for file in all_file:
        pattern = re.compile("../data/ADFA-LD/Attack_Data_Master/Web_Shell_\d+/UAD-W*")
        if re.match(pattern,file):
            with open(file,'r') as f:
                x.append(f.read())
                y.append(1)
    return x,y

def get_feature():
    x1,y1 = get_noranl_data("../data/ADFA-LD/Training_Data_Master/")
    x2,y2 = get_attack_data("../data/ADFA-LD/Attack_Data_Master/")
    x = x1+x2
    Y = y1+y2

    vectorizer = CountVectorizer(min_df=1)  ###将词集向量化
    X = vectorizer.fit_transform(x).toarray()
    return X,Y

def main():
    X,Y = get_feature()
    neigh = KNeighborsClassifier(n_neighbors=3)
    score = cross_val_score(neigh,X,Y,cv=10,n_jobs=1)
    print(score)
main()



