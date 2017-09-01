import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tf_utils import random_mini_batches, predict
from dataset.cifar10 import load_training_data, load_test_data

X_train_orig, Y_train_orig, ohe = load_training_data()
#print("Training set size: " + str(len(X_train_orig)))


def show_random_images():
    index = 0
    fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X_train_orig)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X_train_orig[i:i + 1][0])
    plt.show()

#show_random_images()

# Flatten the training images
X_train_all = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_train = X_train_all[:, :10000]
Y_train_all = ohe.T
Y_train = Y_train_all[:, :10000]
print("X_train_orig shape: " + str(X_train_orig.shape))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))

def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype = "float32", shape = [n_x, None])
    Y = tf.placeholder(dtype = "float32", shape = [n_y, None])

    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [35, 3072], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [35, 1], initializer= tf.zeros_initializer())
    W2 = tf.get_variable("W2", [25, 35], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [25, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [12, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [10, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable("b4", [10, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}
    return parameters

### TEST
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("W3 = " + str(parameters["W3"]))
    print("b3 = " + str(parameters["b3"]))
    print("W4 = " + str(parameters["W4"]))
    print("b4 = " + str(parameters["b4"]))

def forward_propagation(X, parameters):

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)

    Z4 = tf.add(tf.matmul(W4, A3), b4)

    return Z4

### TEST
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(3072, 10)
    parameters = initialize_parameters()
    print("X = " + str(X))
    print("Y = " + str(Y))
    Z4 = forward_propagation(X, parameters)
    print("Z4 = " + str(Z4))


def compute_cost(Z4, Y):
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

### TEST
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(3072, 10)
    parameters = initialize_parameters()
    Z4 = forward_propagation(X, parameters)
    cost = compute_cost(Z4, Y)
    print("cost = " + str(cost))

def model(X_train, Y_train, learning_rate = 0.005, num_epochs = 1500, minibatch_size = 128, print_cost = True):
    ops.reset_default_graph()

    tf.set_random_seed(1)

    seed = 3

    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z4 = forward_propagation(X, parameters)

    cost = compute_cost(Z4, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = minibatch_size, seed = seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate = " + str(learning_rate))
        # plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy: ", accuracy.eval({X: X_train, Y: Y_train}))

        return parameters

parameters = model(X_train, Y_train)