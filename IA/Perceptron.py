#-*-coding:utf8;-*-
#qpy:3
#qpy:console



import math as mt
import pickle
      
def perceptron(z) :
    if z > 0 :
        return 1
    else :
        return 0

def sigmoid(z) :
    if z > 500 :
        return 1
    elif z<-500 :
        return 0
    else :
        return 1/(1+mt.exp(-z))

class Neuron :
    def __init__(self, size) :
        self.bias = 1
        self.weights = [1 for i in range(0,size)]

    def add(self, x) :
        if len(x) != len(self.weights) :
            print("dimension error", len(x), len(self.weights) )
        yin = 0
        for i,ai in enumerate(x) :
            yin += ai*self.weights[i]
        yin += self.bias
        return yin
  
            
    def output(self, x, fonction=sigmoid ) :
        return fonction(self.add(x))

    def delta(self, x, t, alpha) :
        return self.correction(x, self.output(x), t, alpha)
        
    def correction(self, x, y, t, alpha) :
        if len(x) != len(self.weights) :
            print("dimension error", len(x), len(self.weights))
        self.bias += alpha*(t-y)
        for i, ai in enumerate(x) :
            self.weights[i] += alpha*ai*(t-y)
        #print(y)
        return y==t
     
class Network :
    def __init__(self, layers) :
        self.neurons = [ [ Neuron(layers[i-1]) for j in range(layers[i]) ] for i in range(1,len(layers)) ]
        
    def output(self, x) :
        y = [x]
        for c in self.neurons :
          #  print(y)
            newY = []
            for n in c :
                newY.append(n.output(y[-1]))
            y.append(newY)
        return y
        
    def delta(self, x, t, alpha) :
        y = self.output(x)
        #print(y)
        for i,ai in enumerate(self.neurons[-1]) :
            ai.correction(y[-2], y[-1][i], t[i] , alpha)
        for i,ai in enumerate(self.neurons[:-1]) :
            for j,aj in enumerate(ai) :
                for k,ak in enumerate(t) :
                   # if i >0 :
                        #print(i,j,k)                      
                        aj.correction(y[i], y[-1][k], ak, alpha)
       # print(y)
        return y[-1]==t
        
# p = Neuron(3)
# print(p.output([0,-0.5,0]))
# 
# 
# net = Network([3,4,3,2])
#print(net.output([0,-532,0]))

#while net.delta([0,1,0], [0.3,.8], 0.01) != True :
 #   pass
 
 
#  
# f = open("t10k-labels.idx1-ubyte", "rb")
# #mon_depickler = pickle.Unpickler(testLabel)
# #score_recupere = mon_depickler.load()
# #print(pickle.load(f))
# f.close()
# 
# f2 = open("t10k-images.idx3-ubyte", "rb")
# #mon_depickler = pickle.Unpickler(testLabel)
# #score_recupere = mon_depickler.load()
# #print(pickle.load(f))
# print(f2.read())
# f.close()

import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# training_data, validation_data, test_data = load_data_wrapper()
# training_data = list(training_data)

training_data = load_data()[0]

# net = Network([784,20,10,15,10])
# #print(net.output([0,-532,0]))
# 
# for i in range(100) :
#     t = [0 for i in range(10)]
#     t[training_data[1][i]] = 1
#     net.delta(training_data[0][i], t,  0.1)
#     
# print(net.output(training_data[0][20000]), training_data[1][20000])

# net.delta(training_data[0][0], [0,0,0,0,0,1,0,0,0,0],  .1)
# print(net.output(training_data[0][1]))

# n = Network([784,20,15,10])
# tot =0
# #n=Neuron(784) 
# for i in range(100) :
#     #if training_data[1][i] == 0 or training_data[1][i] == 1 :
#     t = [0 for i in range(10)]
#     t[training_data[1][i]] = 1
#     n.delta(training_data[0][i], t,  .1)
# 
# r = 0
# tot = 0
# e = 0
# a=0
# b=0
# c=0
# d=0
# for i in range(50) :
#     t = training_data[1][i+10000]
#     #if t == 0 or t == 1 :
#     y  = n.output(training_data[0][i+10000])[-1]
#     for j,aj in enumerate(y) :
#         if aj == max(y) :
#             break
#     
#     # if out == t :
#     #     r +=1
#     # if out != 1 and out != 0 :
#     #     e += 1
#     
#     # if out > 0.5 :
#     #     if t > 0.5 :
#     #         a+=1
#     #     else :
#     #         b+=1
#     # else :
#     #     if t > 0.5 :
#     #         c+=1
#     #     else :
#     #         d+=1
#     
#     if j == t :
#         a+=1
#     else :
#         b += 1
#         
#     tot += 1
# print(r, tot, e)
# print(a,b,c,d)


n = Network([784,20,1])
tot = 0
#n=Neuron(784) 
alpha = 5
for i in range(1000) :
    #if training_data[1][i] == 0 or training_data[1][i] == 1 :
    t = [training_data[1][i] % 2 ]
    n.delta(training_data[0][i], t, alpha)
    alpha /= 2
    print(i)

print("\n-------------\n")

r = 0
tot = 0
e = 0
a=0
b=0
c=0
d=0
for i in range(100) :
    t = training_data[1][i+10000] % 2
    if t == 0 or t == 1 :
        y  = n.output(training_data[0][i+10000])[-1][0]
        #for j,aj in enumerate(y) :
         #   if aj == max(y) :
          #      break
        
        # if out == t :
        #     r +=1
        # if out != 1 and out != 0 :
        #     e += 1
        
        if y > 0.5 :
            if t > 0.5 :
                a+=1
            else :
                b+=1
        else :
            if t > 0.5 :
                c+=1
            else :
                d+=1
        
        # if j == t :
        #     a+=1
        # else :
        #     b += 1
        # 
        tot += 1
        print(i)
print(r, tot, e)
print(a,b,c,d)


#print(len(load_data()[0][1]))