import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def get_one_file(filename):
    string = ""
    with open(filename,'r') as f:
        for line in f.readlines():
            string += line.strip()
    return string

def convert_to_string(rootdir):
    file_list = os.listdir(rootdir)
    x = []
    for file in file_list:
        if file.endswith('.php'):
            file_path = os.path.join(rootdir,file)
            x.append(get_one_file(file_path))
    return x

def webshell_vectorize():
    wb_vectorizer = CountVectorizer(ngram_range=(2,2),decode_error='ignore',token_pattern=r'\bw+\b',min_df=1)
    wb_file_list = convert_to_string("../data/PHP-WEBSHELL/dama/")
    x1 = wb_vectorizer.fit_transform(wb_file_list).toarray()
    vocabulary = wb_vectorizer.vocabulary_
    return x1,vocabulary

def wbshell_vectorize_new():
    wbshell_vectorizer = CountVectorizer(ngram_range=(1,1),decode_error='ignore',token_pattern=r'\b\w+\b\(|\'\w+\'',min_df=1)
    wb_file_list = convert_to_string("../data/PHP-WEBSHELL/dama/")
    x1 = wbshell_vectorizer.fit_transform(wb_file_list).toarray()
    return x1,vocabulary

def get_feature_new():
    x1,vocabulary = wbshell_vectorize_new()
    wp_vectorizer = CountVectorizer(ngram_range=(1,1),decode_error='ignore',token_pattern=r'\b\w+\b\(|\'\w+\'',min_df=1,vocabulary=vocabulary)
    wp_file_list = convert_to_string("../data/wordpress")
    x2 = wp_vectorizer.fit_transform(wp_file_list).toarray()
    y1, y2 = np.ones((x1.shape[0], 1)), np.zeros((x2.shape[0], 1))

    X = np.vstack((x1, x2))
    Y = np.vstack((y1, y2)).ravel()
    return X, Y

def get_feature():
    x1,vocabulary = webshell_vectorize()
    wp_vectorizer = CountVectorizer(ngram_range=(2,2),decode_error='ignore',token_pattern=r'\bw+\b',min_df=1,vocabulary=vocabulary)
    wp_file_list = convert_to_string("../data/wordpress")
    x2 = wp_vectorizer.fit_transform(wp_file_list).toarray()

    y1,y2 = np.ones((x1.shape[0],1)),np.zeros((x2.shape[0],1))

    X = np.vstack((x1,x2))
    Y = np.vstack((y1,y2)).ravel()
    return X,Y

def main():
    X,Y = get_feature()
    clf = GaussianNB()
    score = cross_val_score(clf,X,Y,cv=3)

    print("score=",score)
def main_1():
    X,Y = get_feature_new()
    clf = GaussianNB()
    score = cross_val_score(clf,X,Y,cv=3)

    print("score=",score)
main_1()