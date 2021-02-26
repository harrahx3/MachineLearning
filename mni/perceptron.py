import math as mt
import random as rd

def sign(a) :
    if a > 0 :
        return 1
    else :
        return -1

class Perceptron :
    
    def  __init__ (self, size=2, lr=.1) :
        self.weights = [rd.random()*2-1 for i in range(0, size)]
        self.bias = rd.random()*2-1
        self.lr = lr       #learning rate
        
    def guess (self, inputs, activation=sign) :
        z = 0;
        for i, ai in enumerate(inputs) :
            z += ai * self.weights[i]
        return activation (z + self.bias)
        
    def train(self, inputs, target) :
        guessed = self.guess(inputs)
        error = target - guessed
        
        for i in range(len(self.weights)) :
            self.weights[i] += error * inputs[i] * self.lr
            
        self.bias += error * self.lr