import gzip
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def load_data():
    with gzip.open("../data/mnist.pkl.gz") as f:
        train_data,valid_data,test_data = pickle.load(f,encoding='bytes')
    return train_data,valid_data,test_data

if __name__ == '__main__':
    train_data,valid_data,test_data = load_data()
    x1,y1 = train_data
    x2,y2 = test_data
    clf = LogisticRegression(C=1e5)
    clf.fit(x1,y1)

    score = cross_val_score(clf,x2,y2,scoring="accuracy")

    print("score=",score)
