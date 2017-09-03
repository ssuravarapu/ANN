import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from dataset.cifar10 import load_training_data, load_test_data

X_train, _, Y_train = load_training_data()
print("X_train *original* shape: " + str(X_train.shape))
print("Y_train *original* shape: " + str(Y_train.shape))

X_test, _, Y_test = load_test_data()
print("X_test *original* shape: " + str(X_test.shape))
print("Y_test *original* shape: " + str(Y_test.shape))

def preprocess_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print("X_train *modified* shape: " + str(X_train.shape))
    print("X_test *modified* shape: " + str(X_test.shape))

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def build_model():
    model = Sequential()
    model.add(Dense(500, activation='relu', input_shape=(3072,)))
    model.add(Dropout(0.33))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    print("Model Summary: " + "\n" + str(model.summary()))
    print("Model Config: " + "\n" + str(model.get_config()))
#    print("Model Weights: " + "\n" + str(model.get_weights()))

#    decay = 0.00001

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

model = build_model()

X_train, X_test = preprocess_data(X_train, X_test)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

start = time.time()
model.fit(X_train, Y_train, validation_data=(X_test[:5000], Y_test[:5000]), callbacks=[early_stopping], epochs=200,
          batch_size=128, verbose=2)
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))

y_pred = model.predict(X_test[5000:10000], batch_size=128)
score = model.evaluate(X_test[5000:10000], Y_test[5000:10000], verbose=1)

print(score)