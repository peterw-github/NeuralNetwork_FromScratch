import mnist_loader
import neuralnet
import pandas as pd
import numpy as np
import random



# load data using using mnist_loader
training_data, test_data = mnist_loader.load_data('./Data/mnist_train.csv', './Data/mnist_test.csv')


# Creating a neural network
net = neuralnet.NeuralNetwork([784, 30, 10])


# Calculate the accuracy of the network
acc = net.calcAccuracy(test_data)
print("The models accuracy before any training, is: {}%".format(acc * 100))


# Train the network
batch_size = 10
alpha_v = 10
n_epochs = 1
for i in range(1, 20):
    net.stochastic(training_data, n_epochs, batch_size, alpha_v)
    # Recalculate accuracy
    acc = net.calcAccuracy(test_data)
    print("The models accuracy after {} epoch(s) of training, is: {:.2f}%".format(i, acc*100))


