# -*- coding: utf-8 -*-



import numpy as np_
from scipy.signal import sawtooth
import matplotlib.pyplot as plt_



def Sigmoid (min_,max_,len_, height) :

    # t = np_.linspace(-2.5, 2.5, 50)
    # y = 1.5/(1.0 + np_.exp(-t/0.5))
    
    t = np_.linspace(min_, max_, len_)
    # y = height/(1.0 + np_.exp(-t/speed))
    y = height/(1.0 + np_.exp(-t))
    
    return y



def Sigmoid_Inv(min_,max_,len_, height) :
    
    sigm = Sigmoid (min_,max_,len_, height)
    
    return 1 - sigm


    
def Triangle (min_,max_,len_, height ,width) :
    
    # t = np_.linspace(0, 1, 60)
    # y = (1.5/2.0)*(sawtooth(2 * np_.pi *t, width=0.5) + 1.0)
    
    t = np_.linspace(min_, max_, len_)
    y = height*(sawtooth(2 * np_.pi *t, width=width) + 1.0)
    
    return y

def Rect ():
    
    t = np_.linspace(-1, 1, 50)
    y = [(0 if abs(elem)>0.5 else (0.5 if abs(elem)==0.5 else 1)) for elem in t]
    
    return np_.asanyarray(y)


def Rect_Inv ():
 
    t = np_.linspace(-1, 1, 50)
    y = [(0 if abs(elem)<0.5 else (0.5 if abs(elem)==0.5 else 1)) for elem in t]
    
    return np_.asanyarray(y)


# Mean smooth
# sigm = Sigmoid_Flt(-5, 5, 50, 1.5)
# sigm_inv = Sigmoid_Inv_Flt(-5, 5, 50, 1.5)#(-2.5, 2.5, 50, 1.5, 0.5)
#trig = Triangle_Flt(0, 1, 60, 1.5/2.0, 0.5)

# Median smooth

# sigm = Sigmoid(-100, 100, 25, 1.5)
# sigm_inv = Sigmoid_Inv(-100, 100, 25, 1.5)#(-2.5, 2.5, 50, 1.5, 0.5)
# trig = Triangle(0, 1, 60, 1.5/2.0, 0.5)
# p= Rect()

# plt_.figure()
# plt_.plot(sigm)
# plt_.figure()
# plt_.plot(sigm_inv)
# plt_.figure()
# plt_.plot(p)







# def Sigmoid_Flt () :

#     t = np_.linspace(-2.5, 2.5, 50)
#     y = 1.75/(1.0 + np_.exp(-t/0.5))
    
#     return y


# def Sigmoid_Inv_Flt() :
    
#     sigm = Sigmoid_Flt()
    
#     return 1 - sigm


    
# def Triangle_Flt () :
    
#     t = np_.linspace(0, 1, 60)
#     y = (1.5/2.0)*(sawtooth(2 * np_.pi *t, width=0.5) + 1.0)
    
#     return -y 