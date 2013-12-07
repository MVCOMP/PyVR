import numpy
from sklearn import svm
from sklearn.mixture import GMM
import sys

class EmoModelOneClassClassifier():
    def __init__(self,_type=None,gamma=None):
        self.data = []
        self.model = None
        self.type = _type
        self.gamma = 0.000022 #0.000155555555556
        self.nu= 0.145
        self.attr = [2,7,8,29,34,46,48,52,55,59]

    def train(self,obs):
        obs = numpy.array(obs)
        
        obs = obs[:,self.attr]
        
        num_components = 10
	try:
	        gmm = GMM(n_components=num_components,covariance_type='diag')
	except:
	        gmm = GMM(n_components=num_components,cvtype='diag')

        gmm.fit(obs)
        predictions = gmm.predict(obs)
        
        for n in range(num_components):
            indexes = numpy.where(predictions==n)[0]
            if len(indexes)>2:
                s_obs = obs[indexes]
                self.data.append(s_obs.mean(0))        
        
        X = numpy.array(self.data)
        
        try:
            self.model = svm.OneClassSVM(nu=self.nu,gamma=self.gamma)
            #self.model = svm.OneClassSVM(nu=0.1,gamma=Gamma)
            self.model.fit(X)
        except:
            print "exception in EmoModelOneClassClassifier.train()"
        

    def test(self,fingerprint):
        
        
        #X_test = fingerprint.getMeans(self.attr)
        X_test = numpy.mean(fingerprint.getObs(self.attr))
               
        # X_test = numpy.array(obs.mean(0))
        y_test = self.model.predict(X_test)[0]
        dist = self.model.decision_function(X_test)[0][0]
        return y_test,dist
        

class EmoModelException(Exception):
    def __init__(self, message, Errors):        
        self.Errors = Errors        
        
        
class EmoModel():
    def __init__(self,gamma=None):
        self.classifiers = []   #vector vacio de clasificadores
        
        Neutral = EmoModelOneClassClassifier('NEUTRAL',gamma)
        self.classifiers.append(Neutral)
        
        Angry = EmoModelOneClassClassifier('ANGRY',gamma) 
        self.classifiers.append(Angry)
        
        Happy = EmoModelOneClassClassifier('HAPPY')
        self.classifiers.append(Happy)
        
        Sad = EmoModelOneClassClassifier('SAD') 
        self.classifiers.append(Sad) 
        
        Disgust = EmoModelOneClassClassifier('DISGUST')
        self.classifiers.append(Disgust)
        
        Anxiety_Fear = EmoModelOneClassClassifier('ANXIETY-FEAR')
        self.classifiers.append(Anxiety_Fear)
        
        Bored = EmoModelOneClassClassifier('BORED')
        self.classifiers.append(Bored)
    
    def trainFromStorage(self, storage):
        emos = storage.getAllEmotions()
        for emotion in emos.keys():
            for clf in self.classifiers:
                if clf.type == emotion and emotion != None:
                    obs = emos[emotion]
                    clf.train(obs)


    def test(self,fv):
        results = [('?',numpy.inf)]
        for clf in self.classifiers:
            if clf.data != []:
                result,dist = clf.test(fv)                
                if result == 1:
                    results.append((clf.type,dist))
        results = sorted(results, key=lambda res: res[1])
                
        return results[0]
    
            
