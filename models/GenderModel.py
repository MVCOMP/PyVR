#! /usr/bin/env python
# encoding: utf-8

import numpy
from sklearn.mixture import GMM
from sklearn import preprocessing
from ModGmm import modifiedGMM

class GenderModelException(Exception):
    def __init__(self, message, Errors):
        self.Errors = Errors


class GenderModel():
    
    def __init__(self, trained_model=[]):
        self.trained_model = trained_model
        self.ids = set()
        # What is this list witchcraft?
        self.attr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                     17, 19, 20, 21, 25, 40, 43]

    def train_from_observations(self, observations):
        '''
            Train model, receives an observation dictionary as so
            
            { 
                gender: [obs, obs,...],
                ...
            }
        '''

        # We are using numpy arrays instead of lists
        for one_gender in observations:
            converted = numpy.array(observations[one_gender])
            obs = converted[:,self.attr]
            
            normalizer = preprocessing.Normalizer().fit(obs)
            obs = normalizer.transform(obs)

            gmm = modifiedGMM(5)
            gmm.fit(obs)
            gmm['gender'] = one_gender
            gmm['normalizer'] = normalizer
            self.trained_model.append(gmm)

        #TODO: What is this doing here?
	    '''
            # Bayesian Information Criteria (BIC)
            best_gmm = None
            lowest_bic = numpy.infty
            n_components_range = range(1,7)
            for n_components in n_components_range:
                # Estimate model parameters with the expectation-maximization algorithm.
                gmm = modifiedGMM(n_components)
                gmm.fit(obs)
                # Bayesian information criterion for the current model fit and the proposed data
                bic = gmm.bic(obs)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
            
            best_gmm['gender'] = gender
            best_gmm['normalizer'] = normalizer
            self.trained_model.append(best_gmm)
            '''

        return True
    
    
    def getOrderedList (self, obs):
        r = [(gmm,numpy.sum(gmm.loglikelihood(obs))) for gmm in self.trained_model]
        r.sort(key=lambda ld: ld[1])
        r.reverse()
        return r
    
    def testObs(self, obs):
        obs = obs[:,self.attr]
        labeled_distances = self.getOrderedList(obs)
        nearest_model = labeled_distances[0][0] # GMM instance
        second_model = labeled_distances[1][0] # GMM instance
        value = nearest_model["gender"]
	score = nearest_model.loglikelihood(obs) - second_model.loglikelihood(obs)
        return value, score


    def test(self,fv):
        obs = fv.getObs()
        return self.testObs(obs)    