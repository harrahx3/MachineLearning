import math as mt
import random as rd
import perceptron as p
import Point as pt
import Network as nen
import Matrix as mat

import tkinter as tk
import nn

#cd Documents\projets python\ML
# C:\Users\Hyacinthe\Documents\projets python\DeepLearningPython35-master
brain = p.Perceptron()
net = nen.Network(2,10,1)
print (brain.guess([1,-1]))
#print (net.feedforward([1,-1]))

#n = nn.Nn([2,2,1])

top = tk.Tk()
h = 400
w = 400
C = tk.Canvas(top, bg="white", height=h, width=w)

#trainingPoints = [pt.Point() for i in range(0, 20000)]
trainingPoints = 1000 * [pt.Point.create(0,0,-1),pt.Point.create(0,400,1),pt.Point.create(400,400,-1),pt.Point.create(400,0,1) ]

for point in trainingPoints :
   # brain.train([point.x/w, point.y/h], point.label)
   net.train([point.x/w, point.y/h], [point.label])
   # n.SGD(([point.x/w, point.y/h], [point.label]), 1, 1, 1)

testPoints = [pt.Point() for i in range(0,1000)]

for point in testPoints :
    point.show(C)
    
for point in testPoints :    
    coord = point.x-3, point.y-3, point.x+3, point.y+3
    #if point.label == brain.guess([point.x/w, point.y/h]) :
    if point.label == p.sign(net.feedforward([point.x/w, point.y/h])[0][0] - .5):
        arc = C.create_oval(coord, fill="white")

C.pack()

top.mainloop()
