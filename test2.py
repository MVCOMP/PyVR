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
    scores_ok = load('scores_ok.pkl')
    scores_err = load('scores_err.pkl')
    
    u_ok = numpy.mean(scores_ok)
    u_err = numpy.mean(scores_err)
    
    total = len(_testlist)
    
    thr = .5*(u_ok+u_err)
    
    #x = numpy.linspace(0.0, thr)
    x = numpy.linspace(0.0, 1.0,1000)
    
    # hit
    TP = numpy.array([1.*numpy.sum([1 for s in scores_ok if s > x_])/total for x_ in x])
    
    # false alarm
    FP = numpy.array([1.*numpy.sum([1 for s in scores_err if s > x_])/total for x_ in x])
    
    # correct rejection
    TN = numpy.array([1.*numpy.sum([1 for s in scores_err if s < x_])/total for x_ in x])

    # miss
    FN = numpy.array([1.*numpy.sum([1 for s in scores_ok if s < x_])/total for x_ in x])
    
    
    # sensitivity or true positive rate (TPR) | TPR = TP / ( TP + FN )
    TPR = TP / ( TP + FN )

    # specificity (SPC) or True Negative Rate | SPC = TN / ( TN + FP )
    SPC = TN / ( TN + FP )
    
    # accurancy (ACC) | ACC = ( TP + TN ) / ( TP + TN + FP + FN )
    ACC = ( TP + TN ) / ( TP + TN + FP + FN )

    a = numpy.abs(SPC-ACC)
    idx = a.argmin(axis=0)
    opt_thr = ACC[idx]
    
    print opt_thr
    
    
    plt.subplot(3, 1, 1)
    plt.title('id stats - thresshold (%.3f)'%opt_thr)
    plt.plot(x, TP, 'r')
    plt.plot(x, [opt_thr for n in x] , 'k-')
    #plt.plot(x, [n>idx for n in range(1000)] , 'k-')
    plt.ylabel('TP')

    plt.subplot(3, 1, 2)
    plt.plot(x, FP, 'r')
    plt.plot(x, FN, 'b')
    plt.plot(x, TN, 'g')
    #plt.plot(x, [opt_thr for n in x] , 'k-')
    #plt.plot(x, [n>idx for n in range(1000)] , 'k-')
    plt.ylabel('FP(r) FN(b) TN(g)')


    plt.subplot(3, 1, 3)
    plt.plot(x, TPR, 'g')
    plt.plot(x, SPC, 'r')
    plt.plot(x, ACC, 'b')
    #plt.plot(x, [opt_thr for n in x] , 'k-')
    #plt.plot(x, [n>idx for n in range(1000)] , 'k-')
    print '''
    sensitivity or true positive rate (TPR) | TPR = TP / ( TP + FN )
    specificity (SPC) or True Negative Rate | SPC = TN / ( TN + FP )
    accurancy (ACC) | ACC = ( TP + TN ) / ( TP + TN + FP + FN )    
    '''
    plt.ylabel('TPR(g) SPC(r) ACC(b)')
    
    plt.show()