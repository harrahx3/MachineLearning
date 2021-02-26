#from mnist_loader.py import *
# Third-party libraries

#from pac import my_load_data
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from init import *
#import mnist_loader
# 
# #cd "C:\Users\Hyacinthe\Documents\projets python\mni"
# 
# 
# # %load mnist_loader.py
# """
# mnist_loader
# ~~~~~~~~~~~~
# A library to load the MNIST image data.  For details of the data
# structures that are returned, see the doc strings for ``load_data``
# and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
# function usually called by our neural network code.
# """
# 
# #### Libraries
# # Standard library
# import pickle
# import gzip
# 
# # Third-party libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import random as rd
# 
# def load_data():
#     """Return the MNIST data as a tuple containing the training data,
#     the validation data, and the test data.
#     The ``training_data`` is returned as a tuple with two entries.
#     The first entry contains the actual training images.  This is a
#     numpy ndarray with 50,000 entries.  Each entry is, in turn, a
#     numpy ndarray with 784 values, representing the 28 * 28 = 784
#     pixels in a single MNIST image.
#     The second entry in the ``training_data`` tuple is a numpy ndarray
#     containing 50,000 entries.  Those entries are just the digit
#     values (0...9) for the corresponding images contained in the first
#     entry of the tuple.
#     The ``validation_data`` and ``test_data`` are similar, except
#     each contains only 10,000 images.
#     This is a nice data format, but for use in neural networks it's
#     helpful to modify the format of the ``training_data`` a little.
#     That's done in the wrapper function ``load_data_wrapper()``, see
#     below.
#     """
#     f = gzip.open('mnist.pkl.gz', 'rb')
#     training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
#     f.close()
#     return (training_data, validation_data, test_data)
# 
# def load_data_wrapper():
#     """Return a tuple containing ``(training_data, validation_data,
#     test_data)``. Based on ``load_data``, but the format is more
#     convenient for use in our implementation of neural networks.
#     In particular, ``training_data`` is a list containing 50,000
#     2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
#     containing the input image.  ``y`` is a 10-dimensional
#     numpy.ndarray representing the unit vector corresponding to the
#     correct digit for ``x``.
#     ``validation_data`` and ``test_data`` are lists containing 10,000
#     2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
#     numpy.ndarry containing the input image, and ``y`` is the
#     corresponding classification, i.e., the digit values (integers)
#     corresponding to ``x``.
#     Obviously, this means we're using slightly different formats for
#     the training data and the validation / test data.  These formats
#     turn out to be the most convenient for use in our neural network
#     code."""
#     tr_d, va_d, te_d = load_data()
#     training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
#     training_results = [vectorized_result(y) for y in tr_d[1]]
#     training_data = zip(training_inputs, training_results)
#     validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
#     validation_data = zip(validation_inputs, va_d[1])
#     test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
#     test_data = zip(test_inputs, te_d[1])
#     return (training_data, validation_data, test_data)
# 
# def vectorized_result(j):
#     """Return a 10-dimensional unit vector with a 1.0 in the jth
#     position and zeroes elsewhere.  This is used to convert a digit
#     (0...9) into a corresponding desired output from the neural
#     network."""
#     e = np.zeros((10, 1))
#     e[j] = 1.0
#     return e
# 
# def vectorized_result_inverse(e):
#     for i,ai in enumerate(e) :
#         if ai == 1 :
#             return i
# 
# # def draw(index) :
# #     #index = 5
# #     dim = 28
# #     img = data[2][0][index]
# #
# #     tab = np.zeros((dim,dim))
# #     for line in range(dim) :
# #         tab[line] = img[dim*line : dim*(line+1)]
# #     plt.imshow(tab, cmap='Greys')
# #     plt.show()
# #
# #     label = data[2][1][index]
# #     print(label)
# 
# def my_load_data() :
#     data = load_data()
#     training_data = data[0]
#     validation_data = data[1]
#     test_data = data[2]
#     #data2 = np.zeros((np.shape(training_data[0])[0],))
#     training_data2 = np.array([(training_data[0][i], vectorized_result(training_data[1][i])) for i in range(np.shape(training_data[0])[0])])
#     test_data2 = np.array([(test_data[0][i], vectorized_result(test_data[1][i])) for i in range(np.shape(test_data[0])[0])])
# 
#     return training_data2, test_data2
# 
# def draw(data, index) :
#     dim = 28
#     img = data[index][0]
# 
#     tab = np.zeros((dim,dim))
#     for line in range(dim) :
#         tab[line] = img[dim*line : dim*(line+1)]
#     plt.imshow(tab, cmap='Greys')
# 
#     # label = data[index][1]
#     # print(label)
#     plt.title(vectorized_result_inverse(data[index][1]))
#     plt.show()


#data = my_load_data()

# for i in range(10):
#     plt.figure(i)
#     draw(data,rd.randint(0,np.shape(data)[0]))


import Network2 as nen
import time
def init() :
    net = nen.Network([28*28,100,10])
    return net
trainingData, testData = my_load_data()
net = init()

# easy = []
# hard = []
# for i in range(5):
#     plt.figure(i)
#     #draw(data,rd.randint(0,np.shape(data)[0]))
#     draw(data,i)
#     plt.show()
#     time.sleep(.1)
#     # if input()=="a" : easy.append(i)
#     # else : hard.append(i)
#



# def train(nb, begin=0, data=trainingData) :
#     data = np.array(data)
#     for i in range(begin, begin+nb) :
#         #print(i)
#         if i<data.shape[0] :
#             net.train(data[i][0], data[i][1],1)
#             if nb < 5 :
#                 plt.figure(i)
#                 draw(data, i)

def train(nb, batchSize=10, data=trainingData) :
    data = np.array(data)

    for i in range(nb) :
        datum = [rd.choice(data) for j in range(batchSize)]
        net.train_batch(datum)

def guess(nb, begin=0, data=testData):
    correct = 0
    data = np.array(data)
    for i in range(begin, begin+nb) :
        if i<data.shape[0] :
            out = net.guess(data[i][0])
            #print(out)
            for j,aj in enumerate(out) :
                if aj == max(out) :
                    guessed = j
            if guessed == vectorized_result_inverse(data[i][1]) :
                correct += 1
            if nb < 5 :
                print(guessed)
                plt.figure(i)
                draw(data, i)
    print()
    print(correct," -> ", 100*correct/nb, "%")

# train(5)
# print("m")
# guess(10)
