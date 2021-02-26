import math as mt
import random as rd
#import Matrix as mat
import numpy as np

def sigmoid (z) :
    return 1 / (1 + mt.exp(-z))
    
def d_sigmoid(z) :
    # return mt.exp(-z)/(1+mt.exp(-z))**2
    # return sigmoid(z) * (1 - sigmoid(z))
    return z * (1 - z)
    
sigmoid_elmt = np.vectorize(sigmoid)
d_sigmoid_elmt = np.vectorize(d_sigmoid)

def randTab(x,y,min,max) :
    return  np.array([[rd.random() * (max-min) + min for j in range(y)] for i in range(x)])
    
def vectorized(L) :
    e = np.zeros((np.shape(L)[0], 1))
    for i,ai in enumerate(L) :
        e[i] = ai
    return e

def compose(tab, fct) :
    return np.array([ [ fct(tab[i,j]) for j in range(np.shape(tab)[1])] for i in range(np.shape(tab)[0]) ])

class Network2 :
    def __init__(self, inputs, hidden, outputs) :
        self.lr = 1    # learning rate
        
        self.inputNodes = inputs 
        self.hiddenNodes = hidden
        self.outputNodes = outputs
        
        self.weights_ih = randTab(self.hiddenNodes, self.inputNodes, -1, 1)
        self.weights_ho = randTab(self.outputNodes, self.hiddenNodes, -1, 1)
                
        
        self.bias_h =randTab (self.hiddenNodes, 1, -1, 1)
        self.bias_o = randTab (self.outputNodes, 1, -1, 1)
        
    def feedforward(self, inputs, activation=sigmoid_elmt) :
        input = vectorized(inputs)
        # print(self.weights_ho)

        # print(self.bias_o)
        # print(inputs)
        # print(input)
        # print(self.weights_ih)
        hidden = activation(self.weights_ih.dot(input) + self.bias_h)
        # print (hidden)
        output = activation(self.weights_ho.dot(hidden) + self.bias_o)
        
        return output
        
    def train(self, inputs, targets, lr=0, activation=sigmoid_elmt, der_activation=d_sigmoid_elmt) :
        if lr == 0: lr=self.lr
        input = vectorized(inputs)
        target = vectorized(targets)
        hidden = activation(self.weights_ih.dot(input) + self.bias_h)
        output = activation(self.weights_ho.dot(hidden) + self.bias_o)
        
        error_o = target - output
       
        error_h = self.weights_ho.transpose().dot(error_o)
        
        # print(error_o)
        # print(error_h)
        
        # print(error_o.shape)
        # print(der_activation(output).shape)
        # print(self.bias_o.shape)
        # print(error_h.shape)
        # print(der_activation(hidden).shape)
        # print(self.bias_h.shape)
        
        self.bias_o += lr * error_o * (der_activation(output))
        self.bias_h += lr * error_h * (der_activation(hidden))

        
        self.weights_ho += lr * (error_o * der_activation(output)).dot(hidden.transpose())
       # self.weights_ho += error_o * output.compose(der_activation) * hidden.transpose() * lr
        self.weights_ih += lr * (error_h * der_activation(hidden)).dot(input.transpose())
        
# a=[[1,2,3],[5,36,4]]
# b=[[1,2],[5,36],[2,3]]


class Network :
    def __init__(self, size) :
        self.lr = 1    # learning rate
        
        self.size = size
        
        self.weights = np.array([randTab(self.size[i+1], self.size[i], -1, 1) for i in range(len(size)-1)])
                
        self.bias = np.array([randTab (self.size[i+1], 1, -1, 1) for i in range(len(size)-1)])     
        
    def feedforward(self, inputs, activation_function=sigmoid_elmt) :
        input = vectorized(inputs)
        
        activation = np.array([np.zeros(self.size[i]) for i in range(len(self.size))])
        activation[0] = input
        for i in range(len(self.size)-1) :
            activation[i+1] = activation_function(self.weights[i].dot(activation[i]) + self.bias[i])
        
        return activation
        
    def error_estimate(self, target, output) :
        error = np.array([np.zeros(self.size[i]) for i in range(1, len(self.size))])
        error[-1] = target - output
       
        for i in range(len(error)-2, -1, -1) :
            error[i] = self.weights[i+1].transpose().dot(error[i+1])
            
        return error
        
    def parameters_update(self, activation, error, lr, der_activation_function) :
        for i in range(len(self.bias)) :
            # print(error[i].shape)
            # print(activation[i+1].shape)
            # print(self.bias[i].shape)
            self.bias[i] += lr * error[i] * (der_activation_function(activation[i+1]))
            self.weights[i] += lr * (error[i] * der_activation_function(activation[i+1])).dot(activation[i].transpose())
        
        
    def guess(self, inputs, activation_function=sigmoid_elmt) :
        return self.feedforward(inputs, activation_function)[-1]
        
    def train(self, inputs, targets, lr=0, activation_function=sigmoid_elmt, der_activation_function=d_sigmoid_elmt) :
        if lr == 0: lr=self.lr
        input = vectorized(inputs)
        target = vectorized(targets)
        
        activation = self.feedforward(inputs, activation_function)
        
        # activation = np.array([np.zeros(self.size[i]) for i in range(len(self.size))])
        # activation[0] = input
        # for i in range(len(self.size)-1) :
        #     activation[i+1] = activation_function(self.weights[i].dot(activation[i]) + self.bias[i])
        
        # error = np.array([np.zeros(self.size[i]) for i in range(1, len(self.size))])
        # error[-1] = target - activation[-1]
       
#       #   for i in range(len(error)-2, -1, -1) :
        #     error[i] = self.weights[i+1].transpose().dot(error[i+1])
        
        error = self.error_estimate(target, activation[-1])
        self.parameters_update(activation, error, lr, der_activation_function) 
        
        # for i in range(len(self.bias)) :
        #     self.bias[i] += lr * error[i] * (der_activation_function(activation[i+1]))
        #     self.weights[i] += lr * (error[i] * der_activation_function(activation[i+1])).dot(activation[i].transpose())

    def train_batch(self, data, lr=0, activation_function=sigmoid_elmt, der_activation_function=d_sigmoid_elmt) :
        if lr == 0: lr=self.lr
        # assert len(inputs) == len(targets)
        error_tot = np.array([vectorized(np.zeros(self.size[i])) for i in range(1, len(self.size))])
        # print(error_tot.shape)
        # print(error_tot[0].shape)
        # print(error_tot[1].shape)
        # print(error_tot)
        for datum in data :
            #print(datum)
            input = vectorized(datum[0])
            target = vectorized(datum[1])
            
            activation = self.feedforward(input, activation_function)
            error = self.error_estimate(target, activation[-1])
            error_tot += error
            # print(error.shape)
            # print(error_tot.shape)
            # print(error[0].shape)
            # print(error_tot[0].shape)
            # print(error[1].shape)
            # print(error_tot[1].shape)
            
        self.parameters_update(activation, error_tot / len(data), lr, der_activation_function)
        
    def save(self, file="netSave.txt") :
        f = open(file, 'wb')
        f.close()
        
        














