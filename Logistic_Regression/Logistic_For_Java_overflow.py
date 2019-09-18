import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def load_normal_data(rootdir):
    file_list = os.listdir(rootdir)
    x = []
    for file in file_list:
        cmd = ""
        path = os.path.join(rootdir,file)
        with open(path,'r') as f:
            for line in f.readlines():
                cmd += line.strip()
        x.append(cmd)
    return x

def load_attack_data(rootdir):
    x = []
    file_list = os.listdir(rootdir)
    for file in file_list:
        dir_path = os.path.join(rootdir,file)
        if os.path.isdir(dir_path):
            pattern = re.compile("../data/ADFA-LD/Attack_Data_Master/Java_Meterpreter_\d+")
            if re.match(pattern,dir_path):
                dir_path += "/"
                x += load_attack_data(dir_path)
        else :
            with open(dir_path,'r') as f:
                x.append(f.read().strip())
    return x

def get_feature():
    x1 = load_normal_data("../data/ADFA-LD/Training_Data_Master/")
    x2 = load_attack_data("../data/ADFA-LD/Attack_Data_Master/")
    x = x1+x2
    y = [0]*len(x1)+[1]*len(x2)

    cv = CountVectorizer(decode_error='ignore',min_df=1)
    X = cv.fit_transform(x).toarray()
    return X,y

def main():
    X,Y = get_feature()

    clf = LogisticRegression(C=1e5)

    score = cross_val_score(clf,X,Y,cv=10)
    print("score=",score)
main()
