import numpy as np


def sigmoid(sig):
    return 1 / (1 + np.exp(-sig))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNet:
    """
    Neural Net with:
        2 Inputs
        a hidden layer with 2 Neuron
        an output layer
    Each neuron has same weight and bias.
        w = [0, 1]
        b = 0
    """

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o = Neuron(weights, bias)

    def feedforward(self, feed):
        out_h1 = self.h1.feedforward(feed)
        out_h2 = self.h2.feedforward(feed)

        out_o = self.o.feedforward(np.array([out_h1, out_h2]))

        return out_o


network = OurNeuralNet()
x = np.array([2, 3])
print(network.feedforward(x))
