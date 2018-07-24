from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3),
                 activation = 'relu',
                 input_shape = (28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size = 128,
          epochs = 3,
          verbose = 1,
          callbacks = [keras.callbacks.TensorBoard(log_dir = "drive/deeplearning/log", histogram_freq = 0, write_graph = True, write_images = True)],
          validation_data = (x_test, y_test))

model.fit(x_train, y_train,
          batch_size = 128,
          epochs = 10,
          verbose = 1,
          callbacks = [keras.callbacks.TensorBoard(log_dir = LOG_DIR, histogram_freq = 0, write_graph = True, write_grads = True, batch_size = 128, write_images = True)],
          validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose = 0)

score = model.evaluate(x_test, y_test, verbose = 0)
print(score[1])
