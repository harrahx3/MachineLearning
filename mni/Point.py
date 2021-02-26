import math as mt
import random as rd



class Point :
    
    @staticmethod
    def create(x, y, l) :
        pt = Point()
        pt.x = x
        pt.y = y
        pt.label = l
        return pt
        
    def __init__(self) :
        self.x = rd.random()*400
        self.y = rd.random()*400
        self.label = 0
        if self.fct():
            self.label = 1
        else :
            self.label = -1
        
    def show(self, canvas) :
        coord = self.x-5, self.y-5, self.x+5, self.y+5
        if self.label == 1 :
            arc = canvas.create_oval(coord, fill="red")
        elif self.label == -1 :
            arc = canvas.create_oval(coord, fill="blue")
    
    def fct(self) :
        #return (self.x > 200  and self.y > 200) or (self.x < 200 and self.y < 200)
        #return self.x > .6*self.y +200
        return self.x > 350