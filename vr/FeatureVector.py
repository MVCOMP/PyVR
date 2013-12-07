#! /usr/bin/env python
# encoding: utf-8

import numpy
#import time
import sys
from Algorithms import Algorithms
#from ModGmm import modifiedGMM

class FeatureVectorException(Exception):
    def __init__(self, message, Errors ):
        Exception.__init__(self, message)        
        self.Errors = Errors
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print (exc_type, message, exc_tb.tb_lineno)

class FeatureVector:
    
    def __init__(self):
        self.__vector_size = 20
        self.__buffer_size = 160
        self.__step = self.__buffer_size/2
        self.__buffer = numpy.array([],dtype=numpy.float32)
        self.__obs = []
        self.initValues()
        self.attributes = {}
        #self.VADGMM = modifiedGMM(2)
        #self.__mean = numpy.zeros(3*self.__vector_size,dtype=numpy.float32)
        #self.__n = 0
        
    def getExpectedFrames(self,n_samples):
        return n_samples/self.__step

    # Set initial values:   
    def initValues(self):
        
        # initial thresshold for voice activity detection
        self.threshold = 0.1
        self.__VADthd = 0.
        
        # counter for voiced windows
        self.__VADn = 0.
        
        
        self.__silence_counter = 0

        # Memory for previous values (for delta and delta-delta calculations)
        self.__deltabuffer = [numpy.zeros(self.__vector_size,dtype=numpy.float32)]*5
    
    
    # This is the most important function of the class
    # In this function, each data frame (20 ms of audio)
    # is transformed into a feature vector and stored in
    # an array (self.__obs)
    # The mean value of this vectors are also updated
    # in this function.
    
    def __processFrame(self, window):
        
        
        try:
            fv = Algorithms.frameProcessPy(window,self.__vector_size)

            #fv = numpy.array(Algorithms.frameProcess(window,self.__vector_size),dtype=numpy.float32) # * self.__buffer_size
            
            for x in fv:
                if numpy.isnan(x) or numpy.isinf(x): return
        
            X = self.buffrotate(fv)
            if any(X != 0.):
                d = self.delta()
                d2 = self.delta2()
                v = numpy.concatenate([X,d,d2])
                self.__obs.append(v)
    
                #current_mean = (self.__n * self.__mean + v)/(self.__n + 1)
                #self.__mean = current_mean
                #self.__n += 1
    
        except Exception,e:
            raise FeatureVectorException('Error while processing window:\n'+e.message,0)
            
    # Push new observation into a 5 element buffer
    # and returns central element (the buffer is used
    # to aproximate the first derivative)
    
    def buffrotate(self,X):
        self.__deltabuffer.pop(0)
        self.__deltabuffer.append(X)
        return self.__deltabuffer[2]

    
    # Calculate first derivative of the vector
    
    def delta(self):
        f = self.__deltabuffer[0] - 8*self.__deltabuffer[1] + 8*self.__deltabuffer[3]-self.__deltabuffer[4]
        return f/12.

    
    # Calculate seccond derivative of the vector
    
    def delta2(self):
        f = -self.__deltabuffer[0] + 16*self.__deltabuffer[1] -30*self.__deltabuffer[2] + 16*self.__deltabuffer[3] - self.__deltabuffer[4]
        return f/12.

    
    # Voice Activity Detection
    # Adaptative thresholding method
     
    def VAD(self,_frame):
        frame = numpy.array(_frame)**2
        thd = numpy.min(frame) + numpy.ptp(frame)*self.threshold
        self.__VADthd = (self.__VADn * self.__VADthd + thd)/float(self.__VADn + 1.)
        self.__VADn += 1.
        
        if numpy.mean(frame) <= self.__VADthd:
            self.__silence_counter += 1
        else:
            self.__silence_counter = 0
        
        if self.__silence_counter > 3:
            #self.__VADn = 1. # test
            return False
        
        return True
    
    
    # Push new audio samples into the buffer.
    # This method can be called from an external script
    # but it is also called from self.process eliminating
    # this aditional step
    
    def addSamples(self,data):
        data = numpy.array(Algorithms.ButterworthFilter(data),dtype=numpy.float32)
        self.__buffer = numpy.append(self.__buffer,data)
        result = len(self.__buffer) >= self.__buffer_size
        return result
    
    
    # Pull a portion of the buffer to process
    # (pulled samples are deleted after beeng
    # processed
    
    def getFrame(self):
        window = self.__buffer[:self.__buffer_size]
        self.__buffer = self.__buffer[self.__step:]  
        return window
    
    
    # Method called from external function
    # This method adds new audio samples to the internal
    # buffer and process them in order to generate
    # new feature vectors (and update mean value of
    # the vector collection)
    
    def process(self,data):
        if self.addSamples(data):
            # mientras sea posible voy tomando ventanas
            # del segmento total y voy calculando los
            # coeficientes para cada ventana
            while len(self.__buffer) >= self.__buffer_size:
                # Framing
                window = self.getFrame()
                
                try:
                    if self.VAD(window): # speech frame
                        self.__processFrame(window)
                        
                except Exception, e:
                    raise FeatureVectorException('Exception in process:\n'+e.message,0)
            return True
        else:
            return False

    
    # This method is called to get the complete observation
    # set stored in self.__obs
    # If needed, this method can return a set of "smaller"
    # vectors (only with the provided indexes)
    
    def getObs(self, attr = None):
        obs = numpy.array(self.__obs)
        if attr == None:
            return obs
        obs = obs[:,attr]
        return obs
    
                
    # this method returns the mean values of the vectors
    # calculated on-line during processing
    #def getMeans(self,attr = None):
    #    if attr == None:
    #        return self.__mean
    #    return self.__mean[:,attr]
    

    # this method returns the actual length of each
    # feature vector (Not very usefull since this length
    # is a constant parameter)
                
    def __len__(self):
        return self.__vector_size
    
                
    # This method let the external script to asociate values
    # to this class as it were a standard dictionary.
    # Example:
    # fv_instance["some_key"] = "some value"
                
    def __setitem__(self,key,value):
        self.attributes[key] = value
    
    # Example
    # some_variable = fv_instance["some_key"]
    # some_variable should be equal to  "some value"
                
    def __getitem__(self,key):
        try:
            value = self.attributes[key]
        except:
            value = None
        finally:
            return value

