# Importing NumPy, to implement vectorisation efficiently
import numpy as np

# Importing random, to randomly sample from training data:
import random



def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_prime(Z):
    return np.exp(-Z) / (1.0 + np.exp(-Z))**2

class NeuralNetwork:

    def __init__(self, nn_lyrs):
        """
        :param nn_lyrs (list):
            A list of the neural networks layers. For example, if nn_lyrs = [50, 10, 20, 3], then a NeuralNetwork
            object is created, with an input layer of 50 neurons, a 1st hidden layer of 10 neurons, a 2nd hidden layer
            of 20 neurons, and an output layer of 3 neurons.
        """

        # Store nn_lyrs:
        self.nn_lyrs = nn_lyrs

        # Determine number of neuron layers:
        self.num_lyrs = len(nn_lyrs)

        # Generate weights:
        self.weights = []
        for i in range(1, self.num_lyrs):
            w_matrx = np.random.randn(nn_lyrs[i], nn_lyrs[i-1])
            self.weights.append(w_matrx)

        # Generate biases:
        self.biases = [np.random.randn(i, 1) for i in nn_lyrs[1:]]


    def calculateOutput(self, A):
        """ Returns the output layer of neurons, given the input layer 'A': """

        for i in range(0, self.num_lyrs - 1):
            W = self.weights[i]
            B = self.biases[i]
            A = sigmoid(np.dot(W, A) + B)
        return A



    def backpropagate(self, X, Y):
        """
        Returns the gradient of an individual cost, as two separate lists.

        :param X (ndarray): A single training instance from the training data.
        :param Y (ndarray): The corresponding correct/desired answer, to training instance x.

        :return deriv_w (list), deriv_b (list):
            Partial derivatives of individual cost WRT weights are in deriv_w, WRT biases are in deriv_b.
        """

        # First, figure out what all the Z layers are, as well as the A layers (Z_l and A_l):
        A = X
        A_l = [A]
        Z_l = []

        for i in range(0, self.num_lyrs-1):
            W = self.weights[i]
            B = self.biases[i]
            Z = np.dot(W, A) + B
            Z_l.append(Z)
            A = sigmoid(Z)
            A_l.append(A)


        # Now, we can start calculating the partial derivatives of individual cost, with respect to all weights/biases,
        # starting with those located in the final layer:


        # Creating two lists, to store the partial derivatives of individual cost:
        deriv_b = []
        for b_vectr in self.biases:
            empty_vectr = np.zeros(b_vectr.shape)
            deriv_b.append(empty_vectr.transpose())

        deriv_w = []
        for w_matrx in self.weights:
            empty_matrx = np.zeros(w_matrx.shape)
            deriv_w.append(empty_matrx)


        # Calculating deriv of individual cost, WRT unactivated final layer (Page 19 of explanation, equation 1)
        deriv = (A_l[-1] - Y) * sigmoid_prime(Z_l[-1])
        deriv = deriv.transpose()


        # Calculating deriv of individual cost WRT weights/biases of final layer (Page 28 of explanation, equation 7-8)
        deriv_w[-1] = np.dot(deriv.transpose(), A_l[-2].transpose())
        deriv_b[-1] = deriv

        # Go back through all previous layers,
        for i in range(2, self.num_lyrs):

            # Calculating deriv of individual cost, WRT unactivated layer l (Page 22 of explanation, equation 2)
            deriv = np.dot(deriv, self.weights[-i+1]) * sigmoid_prime(Z_l[-i].transpose())

            # Calculating deriv of individual cost, WRT weights/biases of layer l (Equation 7-8 again, page 28)
            deriv_w[-i] = np.dot(deriv.transpose(), A_l[-i-1].transpose())
            deriv_b[-i] = deriv

        # Deriv of individual cost, WRT to all weights/biases, now being returned
        return(deriv_w, deriv_b)


    def update_weights_biases(self, batch, alpha_v):
        """
        Updates/modifies the weights and biases of the neural network.

        :param batch (list):
            A list of tuples. First element in a tuple is a training instance (ndarray). Second element in a tuple
            is the correct answer for that training instance (ndarray)
        :param alpha_v (float):
            The alpha value (AKA the learning rate).
        """

        # Creating two list, to store the sums of the partial derivatives of all individual costs in the batch:
        deriv_b = []
        for b_vectr in self.biases:
            empty_vectr = np.zeros(b_vectr.shape)
            deriv_b.append(empty_vectr.transpose())

        deriv_w = []
        for w_matrx in self.weights:
            empty_matrx = np.zeros(w_matrx.shape)
            deriv_w.append(empty_matrx)

        # Going through each training instance
        for X, Y in batch:

            # Calculating partial derivatives of this individual cost:
            new_deriv_w, new_deriv_b = self.backpropagate(X, Y)

            # Adding partial derivatives of this individual cost, onto the partial derivatives of the previous
            # individual costs of the batch:
            temp_list = []
            zipped_b = zip(deriv_b, new_deriv_b)
            for a, b in zipped_b:
                temp_list.append(a+b)
            deriv_b = temp_list

            temp_list = []
            zipped_w = zip(deriv_w, new_deriv_w)
            for a, b in zipped_w:
                temp_list.append(a+b)
            deriv_w = temp_list


        # Now that all the partial derivatives of ALL individual costs, for the batch have been calculated,
        # we can average amongst them, to approximate the derivative of OVERALL cost, and thus update the weights/biases
        # of the neural network. (Proof is on page 27, equation 5-6)


        # Zipping together networks weights, with weight partial derivatives, then updating networks weights:
        temp_list = []
        weights_and_dw = zip(self.weights, deriv_w)
        for w, dw in weights_and_dw:
            new_w = w + (-dw * alpha_v)/len(batch)
            temp_list.append(new_w)
        self.weights = temp_list

        # Zipping together networks biases, with bias partial derivatives, then updating networks biases:
        temp_list = []
        # Derivatives of weights were stored as row vectors, so to zip, we must first transpose them to column vectors:
        for i in range(0, len(deriv_b)):
            deriv_b[i] = deriv_b[i].transpose()
        biases_and_db = zip(self.biases, deriv_b)
        for b, db in biases_and_db:
            new_b = b + (-db * alpha_v)/len(batch)
            temp_list.append(new_b)
        self.biases = temp_list



    def stochastic(self, train_data, epochs, batch_size, alpha_v):
        """
        Performs stochastic gradient descent on the neural network (trains the network). If test_data is provided,
        then the accuracy of the neural networks predictions on that test_data, will be printed, at the end of
        every training epoch/lap

        :param train_data (list):
            A list full of tuples. Each tuples first element is a training instance (ndarray), and second element is
            the correct answer that goes with that training instance (ndarray).
        :param epochs (int):
            The number of times the neural network will train off all the instances in train_data, after they've
            been grouped into random batches.
        :param batch_size (int):
            The size of a batch, larger batch sizes means weights/biases are updated more slowly. Very small batch
            sizes can lead to weight/biases being updated, rather inaccurately.
        :param alpha_v (int):
            The alpha value (AKA the learning rate).
        """

        # Start performing training epochs:
        n = len(train_data)
        for i in range(0, epochs):

            # Break data up into batches:
            random.shuffle(train_data)
            batches = [train_data[k:k + batch_size] for k in range(0, n, batch_size)]

            # Apply gradient descent to each batch:
            for batch in batches:
                self.update_weights_biases(batch, alpha_v)



    def calcAccuracy(self, test_data):
        """
        Calculates the percentage accuracy of the model
        :param test_data (list):
            A list full of tuples. Each tuples first element is a test instance (ndarray), and second element is
            the correct answer that goes with that test instance (ndarray).
        :return accuracy (float):
            The number of times the model guessed correctly, divided by the number of guesses
        """

        # Calculate the number of test instances:
        n = len(test_data)
        amt_corrct = 0

        # Go through each test instance, making a prediction for each, and incrementing by 1 when we get one correct
        for test_instnce, corrct_answr in test_data:
            predict = np.argmax(self.calculateOutput(test_instnce))
            if predict == corrct_answr:
                amt_corrct += 1
        return amt_corrct/n













