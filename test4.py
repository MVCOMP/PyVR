#! /usr/bin/python

# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="Mauricio"
__date__ ="$10-nov-2013 16:05:20$"

from utils.functions import *
import numpy
import matplotlib.pyplot as plt


def plotGmm(gmm):
    K,D = numpy.shape(gmm.means_)
    
    plt.figure(gmm['id'])
        
    m = -1.0
    M = 1.0
    '''
    for d in range(20):
        for k in range(gmm.n_components):
            mean = gmm.means_[k][d]
            variance = gmm.covars_[k][d]
            w = gmm.weights_[k]
            sigma = numpy.sqrt(variance)
            m = min([m,mean-3*sigma])
            M = max([M,mean+3*sigma])
    '''
    
    for d in range(10):
        for k in range(gmm.n_components):
            mean = gmm.means_[k][d]
            variance = gmm.covars_[k][d]
            w = gmm.weights_[k]
            sigma = numpy.sqrt(variance)
            x = numpy.linspace(m,M,100)
            g = plt.mlab.normpdf(x,mean,sigma)*w
            plt.subplot(10,2,2*d+1)
            plt.ylabel(d)
            plt.plot(x,g)    

    for d in range(10,20):
        for k in range(gmm.n_components):
            mean = gmm.means_[k][d]
            variance = gmm.covars_[k][d]
            w = gmm.weights_[k]
            sigma = numpy.sqrt(variance)
            x = numpy.linspace(m,M,100)
            g = plt.mlab.normpdf(x,mean,sigma)*w
            plt.subplot(10,2,2*(d-10)+2)
            plt.ylabel(d)
            plt.plot(x,g)    

    mng = plt.get_current_fig_manager()
    try:
        mng.frame.Maximize(True)
    except:    
        #mng.resize(*mng.window.maxsize())
        mng.window.state('zoomed')

if __name__ == "__main__":
    idModel = load('idModel.pkl')    
    
    for n,gmm in enumerate(idModel.trained_model):
        print n,'-',gmm['id']
    
    for n in [0,1,2,3]:
        gmm = idModel.trained_model[n]
        plotGmm(gmm)
        
        

    plt.show()

    