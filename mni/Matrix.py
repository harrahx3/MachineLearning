import math as mt
import random as rd
import numpy as np

class Matrix :
    @staticmethod
    def zeros(x, y) :
        mat = Matrix([[]])
        mat.tab = [[0 for j in range(y)] for i in range(x)]
        mat.height = x
        mat.width = y
        return mat
        
    @staticmethod
    def random(x, y, min, max) :
        mat = Matrix([[]])
        mat.tab = [[rd.random() * (max-min) + min for j in range(y)] for i in range(x)]
        mat.height = x
        mat.width = y
        return mat
        
    @staticmethod
    def vector(list) :
        return Matrix([list]).transpose()
        
    def shape(self) :
        return (self.width, self.height)

    def __init__(self, tab) :
        self.tab = tab
        self.height = len(tab)
        self.width = len(tab[0])
        
    def __str__(self) :
        str = ""
        for i in range(self.height) :
            str += "| "
            for j in range(self.width) :
                str +=  self.tab[i][j].__str__() + " "
            str += "|"
            str += "\n"
        return str
        
    def __add__(self, mat) :
        return Matrix([ [ self.tab[i][j]+mat.tab[i][j] for j in range(self.width)] for i in range(self.height)])
        
    def __sub__(self, mat) :
        return self + (mat*-1)
        
    def __mul__(self, mat) :
        if type(mat) ==  type(self):
            return Matrix( [ [ sum([self.tab[i][k] * mat.tab[k][j] for k in range(self.width)]) for j in range(mat.width)] for i in range(self.height)] )
        elif type(mat) == type(0) or type(mat) == type(1.): 
            return Matrix([ [mat * self.tab[i][j] for j in range(self.width)] for i in range(self.height) ])
            
    def __getitem__(self, i) :
        return self.tab[i]
            
    def compose(self, fct) :
        return Matrix([ [ fct(self.tab[i][j]) for j in range(self.width)] for i in range(self.height) ])
        
    def transpose(self) :
        return Matrix([ [ self.tab[j][i] for j in range(self.height)] for i in range(self.width) ])
        
# mat = Matrix.random(2,1, -1, 1)
# mat2 = Matrix ([[0, 1,3], [4, 0, 5]])
# mat22 = Matrix ([[0, 1,3], [5, 0, 5]])
# mat3 = Matrix.vector([1,0,-1])
# print(mat)
# print(mat2)
# print(mat3)
# #print (mat + mat2)
# print (mat*2)
# print (mat.transpose())
# print(mat22-mat2)

# 
# 
# class Matrix :
#     @staticmethod
#     def zeros(x, y) :
#         mat = Matrix([[]])
#         mat.tab = np.zeros(x,y)
#         mat.height = x
#         mat.width = y
#         return mat
#         
#     @staticmethod
#     def random(x, y, min, max) :
#         mat = Matrix([[]])
#         mat.tab = np.array([[rd.random() * (max-min) + min for j in range(y)] for i in range(x)])
#         mat.height = x
#         mat.width = y
#         return mat
#         
#     @staticmethod
#     def vector(list) :
#         return Matrix([list]).transpose()
#         
#     def shape() :
#         return (self.width, self.height)
# 
#     def __init__(self, tab) :
#         self.tab = np.array(tab)
#         self.height = len(tab)
#         self.width = len(tab[0])
#         
#     def __str__(self) :
#         str = ""
#         for i in range(self.height) :
#             str += "| "
#             for j in range(self.width) :
#                 str +=  self.tab[i,j].__str__() + " "
#             str += "|"
#             str += "\n"
#         return str
#         
#     def __add__(self, mat) :
#         return Matrix(self.tab+mat.tab)
#         
#     def __sub__(self, mat) :
#         return self + (mat*-1)
#         
#     def __mul__(self, mat) :
#         if type(mat) ==  type(self):
#             return Matrix( self.tab.dot(mat.tab) )
#         elif type(mat) == type(0) or type(mat) == type(1.): 
#             return Matrix(mat*self.tab)
#             
#     def __getitem__(self, i) :
#         return self.tab[i]
#             
#     def compose(self, fct) :
#         return Matrix([ [ fct(self.tab[i,j]) for j in range(self.width)] for i in range(self.height) ])
#         
#     def transpose(self) :
#         return Matrix([ [ self.tab[j,i] for j in range(self.height)] for i in range(self.width) ])
#         
# 
# 
# 
# 
# 
# 
# 
# 


