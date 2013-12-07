#! /usr/bin/python

# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="Mauricio"
__date__ ="$10-nov-2013 16:05:20$"



from utils.functions import *
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    

    _testlist = load('testlist.pkl')    


    obs = _testlist[_testlist.keys()[0]]['obs']

    N,D = numpy.shape(obs)
    
    print N
    print D
    
    off = 0
    x = range(N)

    for d in range(D/3):
        v = obs[:,d]
        m = numpy.min(v)
        M = numpy.max(v)
        v = (v-m) / (M-m)
        plt.plot(x,v+off)
        off += 1


    plt.figure()

    off = 0
    x = range(D/3)

    p = numpy.zeros(D/3)
    
    for n in range(N):
        v = obs[n,:][:D/3]
        m = numpy.min(v)
        M = numpy.max(v)
        v = (v-m) / (M-m)
        p+=v
        #plt.plot(x,v+off)
        off += 1

    plt.plot(x,p)
    
    '''
    plt.figure()
    
    for d in range(D/3,2*D/3):
        v = obs[:,d]
        m = numpy.min(v)
        M = numpy.max(v)
        v = (v-m) / (M-m)
        plt.plot(x,v+off)
        off += 1

    plt.figure()
    
    for d in range(2*D/3,D):
        v = obs[:,d]
        m = numpy.min(v)
        M = numpy.max(v)
        v = (v-m) / (M-m)
        plt.plot(x,v+off)
        off += 1

'''

    plt.show()
