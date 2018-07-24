import keras
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

from matplotlib import pyplot as plt
_ = plt.imshow(x_train[0])

# Transfrorm our dataset from having shape (n, width, height) to (n, depth, width, height)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert our data type to float32 and normalize our data values to the range (0, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)
print(y_train[:10])

# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(y_train.shape)

