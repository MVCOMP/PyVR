#! /usr/bin/env python
# encoding: utf-8

# Algorithms class is used as a wraper to all
# functions defined in Algorithms.h and
# implemented in Algorithms.c
# Shared library compiled in "libvr.so"
# C functions must be compiled
# gcc-4.2 -c Algorithms.c
# gcc-4.2 -shared -o libvr.so Algorithms.o

import ctypes
import numpy  
import sys
from os.path import abspath, dirname, join

path = dirname(abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

libA = ctypes.cdll.LoadLibrary(join(path, 'libvr.so'))

class AlgorithmsException(Exception):
    def __init__(self, message, Errors ):
        Exception.__init__(self, message)
        self.Errors = Errors
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print (exc_type, message, exc_tb.tb_lineno)


class Algorithms: 

    # void Algorithms_FrameProcess(float * x, float * CEPScoefs )
    @staticmethod
    def frameProcessPy(x,Q):
        M = 40
        N = len(x) 
        
        # ventana de hamming y normalizacion
        hm = x * numpy.hamming(N)/(2**15)
        
        #preenfasis
        hm2 = hm.copy()
        hm2 = numpy.insert(hm2,0,0)
        xpe = hm2[1:] - 0.95 * hm;
        
        #hn = numpy.hanning(N)
        fft = numpy.fft.fft(xpe)[N/2-1:]        

        # tomo la mitad del fft
        N = N/2+1
        spec = numpy.abs(fft)

        fHz = numpy.arange(N)*4000./N
        fmin = 20;
        fmax = 3600;
        
        #fMel = 2595.*numpy.log10(1.+fHz/700.)

        phi_min =  2595.*numpy.log10(1.+fmin/700.)
        phi_max = 2595.*numpy.log10(1.+fmax/700.)
        delta_phi = (phi_max - phi_min)/M

        phi_center = numpy.array([phi_min+ m*delta_phi for m in range(M+1)])
        fc = 700.*(10.**(phi_center/2595.) - 1.) 
        
        X = numpy.zeros(M)
        
        H = numpy.zeros((N,M))
        for m in range(M):
            fa = fc[m]-fc[m-1]
            fb = fc[m]-fc[m+1]
            A = 0.
            for k,f in enumerate(fHz):
                if fc[m-1]<=f and f<fc[m]:
                    H[k,m] = (f-fc[m-1])/fa
                    A += H[k,m]
                elif fc[m]<=f and f<fc[m+1]: 
                    H[k,m] = (f-fc[m+1])/fb
                    A += H[k,m]
                
                
            s = numpy.sum(spec*H[:,m])
            if s>0: X[m] = numpy.log(s/A)
            
        #dct
        c = numpy.zeros(Q+1)
        for l in range(Q):
            c[l] = numpy.sum( X*numpy.cos(l*numpy.pi*(numpy.arange(M)-.5)/M ) )
    
        return c[1:]
    
        
    
    
    @staticmethod
    def frameProcess(x,Q):
        try:
            mfcc_size = 40
            
            if len(x)==160:
                libA.Algorithms_FrameProcess.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
                libA.Algorithms_FrameProcess.restype = None                
                mfcc_coefs=(ctypes.c_float * mfcc_size)()                
                libA.Algorithms_FrameProcess((ctypes.c_float * len(x))(*x), mfcc_coefs)
                # El primer elemento devuelve en realidad el zcr. Los siguientes 19 elementos son MFCC
                #mfcc_coefs = numpy.asarray(mfcc_coefs,dtype=numpy.float32)[1:Q+1]
            return mfcc_coefs[1:Q+1]
        except Exception,e:
            raise AlgorithmsException('Error in Algorithms.frameProcess:\n'+e.message,0)
                                      
    #EXTERN void Algorithms_ButterworthFilter(float *x)
    @staticmethod
    def ButterworthFilter(data):
        try:
            libA.Algorithms_ButterworthFilter.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float), ctypes.c_uint]
            libA.Algorithms_ButterworthFilter.restype = None
            n = len(data)
            if n<1: raise AlgorithmsException('No hay datos para filtrar:\n',0)
            data_bpf = (ctypes.c_float * n)()
            libA.Algorithms_ButterworthFilter((ctypes.c_float * n)(*data),data_bpf,n)
        except Exception,e:
            raise AlgorithmsException('Error in Algorithms.ButterworthFilter:\n'+e.message,0)
        finally:
            return data_bpf[:]


    '''
    # Calcula la distancia de Bhattacharya entre los modelos GMM_A([ua,sa,wa]) y GMM_B([ub,sb,wb])
    # devuelve el valor de la distancia
    # d: distancia de bhattacharya (float)
    @staticmethod
    def bhattacharyaDistance(gmm_a,gmm_b):
        distance = 0.
        
        for i in range(gmm_a.n_components):
            wa = gmm_a.weights_[i]
            ua = gmm_a.means_[i]
            sa = gmm_a.covars_[i]
            
            for j in range(gmm_b.n_components):
                wb = gmm_b.weights_[i]
                ub = gmm_b.means_[i]
                sb = gmm_b.covars_[i]

                Q = len(ua)
                libA.Algorithms_BhattacharyaDistance.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),\
                                                                 ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint]
                libA.Algorithms_BhattacharyaDistance.restype  =  ctypes.c_float
                ua = (ctypes.c_float * len(ua))(*ua)
                ub = (ctypes.c_float * len(ub))(*ub)
                sa = (ctypes.c_float * len(sa))(*sa)
                sb = (ctypes.c_float * len(sb))(*sb)
                distance += libA.Algorithms_BhattacharyaDistance(ua,ub,sa,sb,Q)*wa*wb

        return distance
    '''

        
