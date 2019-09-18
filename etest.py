import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.fashion_mnist
(x_train,y_train) , (x_test,y_test)  = mnist.load_data()

x_train  = x_train / 255
x_test = x_test / 255
print(x_train[0])

print(x_train.shape,y_train.shape)
plt.imshow(x_train[0])
plt.show()