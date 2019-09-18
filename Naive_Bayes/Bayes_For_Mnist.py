import pickle
import gzip
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def load_data():
    with gzip.open('../data/mnist.pkl.gz') as fp:
        training_data,valid_data,test_data = pickle.load(fp,encoding="bytes")
    return training_data,valid_data,test_data

def main():
    train_data,valid_data,test_data = load_data()

    x1,y1=train_data
    x2,y2 = test_data
    clf = GaussianNB()
    clf.fit(x1,y1)

    score = cross_val_score(clf,x2,y2,scoring="accuracy")
    print("score=",score)
main()