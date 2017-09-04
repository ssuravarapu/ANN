import keras
import time
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train original shape: " + str(x_train.shape))
print("x_test original shape: " + str(x_test.shape))
print("y_train original shape: " + str(y_train.shape))
print("y_test original shape: " + str(y_test.shape))

x_train = x_train.reshape(x_train.shape[0], -1) / 255.
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
print("x_train modified shape: " + str(x_train.shape))
print("x_test modified shape: " + str(x_test.shape))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train modified shape: " + str(y_train.shape))
print("y_test modified shape: " + str(y_test.shape))

model = Sequential()
model.add(Dense(200, activation='relu', input_dim=784))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(num_classes, activation='softmax'))

print("Model Summary: " + "\n" + str(model.summary()))
print("Model Config: " + "\n" + str(model.get_config()))

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

start = time.time()
model.fit(x_train, y_train, validation_data=(x_test[:5000], y_test[:5000]), callbacks=[early_stopping], epochs=epochs,
          batch_size=batch_size, verbose=2)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))

y_pred = model.predict(x_test[5000:10000], batch_size=batch_size)
score = model.evaluate(x_test[5000:10000], y_test[5000:10000], verbose=1)

print(score)