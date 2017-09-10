import keras
import time
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, K, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 200
learning_rate = 0.001
decay = learning_rate / epochs

img_rows, img_cols = 32, 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train original shape: " + str(x_train.shape))
print("x_test original shape: " + str(x_test.shape))
print("y_train original shape: " + str(y_train.shape))
print("y_test original shape: " + str(y_test.shape))

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.33))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print("Model Summary: " + "\n" + str(model.summary()))
print("Model Config: " + "\n" + str(model.get_config()))

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


x_train = x_train / 255.
x_test = x_test / 255.

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train modified shape: " + str(y_train.shape))
print("y_test modified shape: " + str(y_test.shape))


early_stopping = EarlyStopping(monitor='val_loss', patience=2)

start = time.time()
model.fit(x_train, y_train, validation_data=(x_test[:5000], y_test[:5000]), callbacks=[early_stopping], epochs=epochs,
          batch_size=batch_size, verbose=2)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))

y_pred = model.predict(x_test[5000:10000], batch_size=batch_size)
score = model.evaluate(x_test[5000:10000], y_test[5000:10000], verbose=1)

print(score)