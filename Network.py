import math as mt
import random as rd
import Matrix as mat

def sigmoid (z) :
    return 1 / (1 + mt.exp(-z))
    
def d_sigmoid(z) :
    # return mt.exp(-z)/(1+mt.exp(-z))**2
    # return sigmoid(z) * (1 - sigmoid(z))
    return z * (1 - z)

class Network :
    def __init__(self, inputs, hidden, outputs) :
        self.lr = 1    # learning rate
        
        self.inputNodes = inputs 
        self.hiddenNodes = hidden
        self.outputNodes = outputs
        
        self.weights_ih = mat.Matrix.random(self.hiddenNodes, self.inputNodes, -1, 1)
        self.weights_ho = mat.Matrix.random(self.outputNodes, self.hiddenNodes, -1, 1)
        
        self.bias_h = mat.Matrix.random(self.hiddenNodes, 1, -1, 1)
        self.bias_o = mat.Matrix.random(self.outputNodes, 1, -1, 1)
        
    def feedforward(self, inputs, activation=sigmoid) :
        input = mat.Matrix.vector(inputs)
        # print(self.weights_ho)

        # print(self.bias_o)
        hidden = (self.weights_ih * input + self.bias_h).compose(activation)
        # print (hidden)
        output = (self.weights_ho * hidden + self.bias_o).compose(activation)
        
        return output
        
    def train(self, inputs, targets, activation=sigmoid, der_activation=d_sigmoid) :
        input = mat.Matrix.vector(inputs)
        target = mat.Matrix.vector(targets)
        hidden = (self.weights_ih * input + self.bias_h).compose(activation)
        output = (self.weights_ho * hidden + self.bias_o).compose(activation)
        
        error_o = target - output
       
        error_h = self.weights_ho.transpose() * error_o
        
        # print(error_o)
        # print(error_h)
        
        self.weights_ho += error_o * output.compose(der_activation) * hidden.transpose() * self.lr
        self.weights_ih += error_h * hidden.compose(der_activation) * input.transpose() * self.lr
        
        self.bias_o += error_o * output.compose(der_activation) * self.lr
        self.bias_h += error_h * hidden.compose(der_activation) * self.lr