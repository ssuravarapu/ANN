import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

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
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    print("Model Summary: " + "\n" + str(model.summary()))
    print("Model Config: " + "\n" + str(model.get_config()))
#    print("Model Weights: " + "\n" + str(model.get_weights()))

    decay = 0.00001

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=decay), metrics=['accuracy'])
    return model

model = build_model()

X_train, X_test = preprocess_data(X_train, X_test)

model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=2)

y_pred = model.predict(X_test, batch_size=128)
score = model.evaluate(X_test, Y_test, verbose=1)

print(score)