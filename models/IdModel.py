#! /usr/bin/env python
# encoding: utf-8

import numpy
#from sklearn.mixture import GMM
from ModGmm import modifiedGMM
from sklearn import preprocessing

class IdModelException(Exception):
    def __init__(self, message, Errors):
        self.Errors = Errors

class IdModel():
    
    def __init__(self, trained_model=[]):
        self.trained_model = trained_model
        self.ids = set()
        self.attr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,25,40,43]

    # Remove storage access from this file, storage should never
    # be accessed from within the lib. Receives ids_dict as previously 
    # returned by Storage.getAllids
    def train_ids(self, ids):
        #ids = storage.getAllIds()
        for k in ids.keys():
            obs = ids[k][:,self.attr]
            normalizer = preprocessing.Normalizer().fit(obs)
            obs = normalizer.transform(obs)
            gmm = modifiedGMM(5)
            gmm.fit(obs)
            gmm['id'] = k
            gmm['normalizer'] = normalizer            
            self.trained_model.append(gmm)
        return True
   
    def verify(self,speakerX,id):
        obs = speakerX.getObs(self.attr)
	ubm = numpy.mean([gmm.loglikelihood(obs) for gmm in self.trained_model if gmm["id"] != id ])
	p = numpy.mean([gmm.loglikelihood(obs) for gmm in self.trained_model if gmm["id"] == id ])
	return p - ubm

    def getOrderedList(self,obs):
        r = [(gmm,gmm.loglikelihood(obs)) for gmm in self.trained_model]
        #r = [(gmm,numpy.sum(gmm.score(gmm['normalizer'].transform(obs)))) for gmm in self.trained_model]
        r.sort(key=lambda ld: ld[1])
        r.reverse()
        return r[:5]

    def testObs(self,obs):
        obs = obs[:,self.attr]
        labeled_distances = self.getOrderedList(obs)
        nearest_model = labeled_distances[0][0] # GMM instance
        max = labeled_distances[0][1] # nearest_model.loglikelihood(obs)
        seccond = labeled_distances[1][1] # second_model.loglikelihood(obs)
        mean = numpy.mean([ld[1] for ld in labeled_distances])
        score = (max-seccond)/(max-mean) 
        value = nearest_model["id"]
        return value,score


    def test(self,fv):
        obs = fv.getObs()
        return self.testObs(obs)    

