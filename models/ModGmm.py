#! /usr/bin/env python
# encoding: utf-8

from sklearn.mixture import GMM
import numpy

class modifiedGMM(GMM):
    def __init__(self,n_components, covariance_type ='diag'):
        try:
		GMM.__init__(self, n_components=n_components, covariance_type=covariance_type)
	except:		
		GMM.__init__(self, n_components=n_components, cvtype=covariance_type)

        self.attributes = {}
    
    def __setitem__(self,key,value):
        self.attributes[key] = value
    
    def __getitem__(self,key):
        try:
            value = self.attributes[key]
        except:
            value = None
        finally:
            return value


    def loglikelihood(self,obs):
	X = self['normalizer'].transform(obs)
	s = self.covars_
	u = self.means_
	w = self.weights_
	
        # N observaciones (c/obs es un vector mfcc+delta+delta2)
        # Las observaciones son vectores de dimension D
	N,D = numpy.shape(X)	
        cm = numpy.sqrt(numpy.prod(s,1)*(2.*numpy.pi)**D)
        ll = numpy.sum([numpy.log(numpy.sum(w * \
        numpy.exp( -numpy.sum( (x-u)**2. / s, 1) ) / cm)) for x in X])/N
        return ll
	
