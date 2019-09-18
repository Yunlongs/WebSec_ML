import matplotlib.pyplot as plt
import gzip
import pickle
from sklearn.neural_network import MLPClassifier

def load_data():
    with gzip.open("../data/mnist.pkl.gz") as f:
        train_data,valid_data,test_data = pickle.load(f,encoding="bytes")
    return train_data,valid_data,test_data

train_data,valid_data,test_data = load_data()
X_train,Y_train = train_data
X_test,Y_test = test_data

mlp = MLPClassifier(hidden_layer_sizes=(50,),max_iter=10,alpha=1e-4,solver='sgd',verbose=10,tol=1e-4,random_state=1,learning_rate_init=.1)

mlp.fit(X_train,Y_train)
print(mlp.score(X_train,Y_train))
print(mlp.score(X_test,Y_test))
