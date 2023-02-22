# A Neural Network, From Scratch

<br>

Hello!

The purpose of this project, was to better understand machine learning, and neural networks in general. 

So, to achieve this, I've coded a neural network (specifically, a multi-layer perceptron), from scratch in Python. 

To help keep the code tidy and presentable, I've created a class specifically designed for building the neural network, and made methods for all of the common calculations (such as calculating gradient, updating weights/biases, calculating output, etc). 

I've also gone through the underlying mathematics of a neural network, which mainly involves multivariable chain rule, vector/matrix calculus, and vectorisation.

Once the neural network was created, it was applied to the MNIST problem, and achieved an accuracy of about 95%, after 10 epochs of training (took around 1 minute of training on my bog standard home computer). 

There are quite a number of improvements that can be made to the neural network that I've built, to further improve accuracy, but since the main goal was just to  improve my understanding of neural networks, by rigorously going through the underlying mathematics, I've decied to leave it as is, and move onto other projects.


<br>
<br>
<br>


## How the code works:

**"main.py"** is a simple script file, that first imports **"mnist_loader.py"** and **"neuralnet.py"**. It then creates a neural network using **"neuralnet.py"**, which is then trained, and the accuracy of the network is printed at the end of every epoch of training, to help demonstrate that the network is learning.

**"mnist_loader.py"**, is a simple program, that loads and transforms the MNIST data into a usable form. 

**"neuralnet.py"**, is where the neural network class has been created, as well as all the methods needed for training and evaulation.


<br>
<br>
<br>


## Walkthrough of underlying mathematics

An explanation/walkthrough of the underlying mathematics can be found here here: <br />
https://github.com/peterw-github/NeuralNetwork_FromScratch/blob/main/Math%20Explanation%20.pdf




<br>
<br>
<br>


## Credit

Credit to Credit to 3Blue1Brown, for the intuitive explanation of a Neural Network, found in the playlist here: <br>
https://www.youtube.com/watch?v=aircAruvnKk

And to Michael Nielsen, for a more indepth explanation on the underlying mathematics of a Neural Network, and recommendations for future improvements, such as 
changing the cost function (currently mean-squared-error method) to a logarithm based one (cross-entropy), as well as exploring other activation functions, such as 
tanh, softmax, and rectified linear unit (ReLU).

Dataset was provided via Kaggle, here: <br>
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

