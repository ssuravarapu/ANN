import keras
import time
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, K
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train original shape: " + str(x_train.shape))
print("x_test original shape: " + str(x_test.shape))
print("y_train original shape: " + str(y_train.shape))
print("y_test original shape: " + str(y_test.shape))

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("x_train modified shape: " + str(x_train.shape))
print("x_test modified shape: " + str(x_test.shape))
print("input_shape: " + str(input_shape))

x_train = x_train / 255.
x_test = x_test / 255.

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train modified shape: " + str(y_train.shape))
print("y_test modified shape: " + str(y_test.shape))

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50, activation='relu', input_dim=784))
model.add(Dropout(0.25))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(num_classes, activation='softmax'))

print("Model Summary: " + "\n" + str(model.summary()))
print("Model Config: " + "\n" + str(model.get_config()))

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

start = time.time()
model.fit(x_train, y_train, validation_data=(x_test[:5000], y_test[:5000]), callbacks=[early_stopping], epochs=epochs,
          batch_size=batch_size, verbose=2)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))

y_pred = model.predict(x_test[5000:10000], batch_size=batch_size)
score = model.evaluate(x_test[5000:10000], y_test[5000:10000], verbose=1)

print(score)