import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


def read_wine_data():
    # White wine data [4898 rows x 12 columns]
    white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                        sep=';')

    # Red wine data [1599 rows x 12 columns]
    red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                      sep=';')

    return white, red


def check_data_quality():
    white, red = read_wine_data()
    print("First rows of White: " + "\n" + str(white.head()))
    print("First rows of Red: " + "\n" + str(red.head()))
    print("Last rows of White: " + "\n" + str(white.tail()))
    print("Last rows of Red: " + "\n" + str(red.tail()))
    print("Sample White: " + "\n" + str(white.sample(5)))  # random sample
    print("Sample Red: " + "\n" + str(red.sample(5)))  # random sample
    print("White description: " + "\n" + str(white.describe()))
    print("Red description: " + "\n" + str(red.describe()))
    print("Null values in White: " + "\n" + str(pd.isnull(white)))
    print("Null values in Red: " + "\n" + str(pd.isnull(red)))


white, red = read_wine_data()

#####
## Exploratory data analysis
#####
def plot_wine_by_alcohol_vol():
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label='Red Wine')
    ax[1].hist(white.alcohol, 10, facecolor='white', ec='black', lw=0.5, alpha=0.5, label='White Wine')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
    ax[0].set_ylim([0, 1000])
    ax[0].set_xlabel("Alcohol in % Vol")
    ax[0].set_ylabel("Frequency")
    ax[1].set_ylim([0, 1000])
    ax[1].set_xlabel("Alcohol in % Vol")
    ax[1].set_ylabel("Frequency")
    fig.suptitle("Distribution of Alcohol in % Vol")
    plt.tight_layout()
    plt.show()


def plot_wine_by_sulphates():
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(red['quality'], red["sulphates"], color="red")
    ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)
    ax[0].set_title("Red Wine")
    ax[1].set_title("White Wine")
    ax[0].set_xlabel("Quality")
    ax[1].set_xlabel("Quality")
    ax[0].set_ylabel("Sulphates")
    ax[1].set_ylabel("Sulphates")
    ax[0].set_xlim([0, 10])
    ax[1].set_xlim([0, 10])
    ax[0].set_ylim([0, 2.5])
    ax[1].set_ylim([0, 2.5])
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Wine Quality by Amount of Sulphates")
    plt.show()


def plot_wine_by_acidity():
    np.random.seed(570)
    redlabels = np.unique(red['quality'])
    whitelabels = np.unique(white['quality'])
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    redcolors = np.random.rand(6, 4)
    whitecolors = np.append(redcolors, np.random.rand(1, 4), axis=0)
    for i in range(len(redcolors)):
        redy = red['alcohol'][red.quality == redlabels[i]]
        redx = red['volatile acidity'][red.quality == redlabels[i]]
        ax[0].scatter(redx, redy, c=redcolors[i])
    for i in range(len(whitecolors)):
        whitey = white['alcohol'][white.quality == whitelabels[i]]
        whitex = white['volatile acidity'][white.quality == whitelabels[i]]
        ax[1].scatter(whitex, whitey, c=whitecolors[i])
    ax[0].set_title("Red Wine")
    ax[1].set_title("White Wine")
    ax[0].set_xlim([0, 1.7])
    ax[1].set_xlim([0, 1.7])
    ax[0].set_ylim([5, 15.5])
    ax[1].set_ylim([5, 15.5])
    ax[0].set_xlabel("Volatile Acidity")
    ax[0].set_ylabel("Alcohol")
    ax[1].set_xlabel("Volatile Acidity")
    ax[1].set_ylabel("Alcohol")
    # ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
    ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
    # fig.suptitle("Alcohol - Volatile Acidity")
    fig.subplots_adjust(top=0.85, wspace=0.7)
    plt.show()

####
## End of exploratory data analysis
####

def preprocess_data():
    red['type'] = 1
    white['type'] = 0
    wines = red.append(white, ignore_index=True)
    print(wines.info())

    # corr = wines.corr()
    # sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    # sns.plt.show()

    X = wines.ix[:, 0:11]
    Y = np.ravel(wines.type)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test

def build_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(11,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print("Model Summary: " + "\n" + str(model.summary()))
    print("Model Config: " + "\n" + str(model.get_config()))
    print("Model Weights: " + "\n" + str(model.get_weights()))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X_train, Y_train, X_test, Y_test = preprocess_data()

model = build_model()

model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=2)

y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test, verbose=1)

print(score)