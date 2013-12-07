#! /usr/bin/python

# To change this template, choose Tools | Templates
# and open the template in the editor.


##
##
##  Se entrena con los primeros 10 segundos de cada archivo
##  luego se crean Test con los siguientes tramos de 3 segundos de cada archivo
##

__author__="Mauricio"
__date__ ="$04-nov-2013 21:21:20$"


from vr.conf import vr_path
from vr.FeatureVector import FeatureVector

from models.IdModel import IdModel

from utils.functions import *
from glob import glob
from os.path import basename

import time

def procFV(samples):
    start = time.clock()
    fv = FeatureVector()
    fv.process(samples)
    elapsed = (time.clock() - start) 
    
    d = {'time':elapsed}
    d['obs'] = fv.getObs()
    
    #l1 = len(samples)
    #l2 = len(fv.getObs())
    #l3 = fv.getExpectedFrames(l1)
    #print 'samples: %i, obs: %i, expected: %i - %.1f%%'%(l1,l2,l3,100.*l2/l3,)

    del(fv)
    return d

    
def readFiles():
    filelist = glob(vr_path+'\\*.wav')
    
    num_processed_files = 0
    total_files = len(filelist)
    
    trainlist = {}
    testlist = {}
     
    for f in filelist:
        filename = basename(f)
        print "Processing %s" % filename
        
        samples = extract_audio_samples(f)

        N = len(samples)
        
        # conservo solamente los primeros 10 segundos        
        # calculo numero de muestras del segmento de prueba
        fr = 8000
        L = fr*15 
        
        if L < N:
            train_samples = samples[:L]
            trainlist[filename] = procFV(train_samples)

            #uso los sigientes tramos de 3 segundos para testear
            M = fr*5
            i=1
            while (L+M*i<N):
                test_samples = samples[L+M*(i-1):L+M*i]
                testlist[filename+'.'+str(i)] = procFV(test_samples)
                i += 1

            num_processed_files += 1
            print '%i - 1 train, %i tests' % (num_processed_files ,i-1)
            
        print "%i/%i OK \n" % (num_processed_files,total_files)


    save(trainlist,'trainlist.pkl')
    save(testlist,'testlist.pkl')

def check(test,result):
    return (test.split('-')[0] == result.split('-')[0])


if __name__ == "__main__":
    # leo todos los wav y genero conjuntos de observaciones para 
    # test y para train. 
    # Se generan los archivos trainlist.pk y testlist.pkl 
    readFiles()    

    # ---------------------------------------------------------

    # creo instancia de modelo de ID
    idModel = IdModel()

    # entreno el modelo (internamente se crean GMM de cada 
    # paquete de observaciones y se almacenan en un array)
    _trainlist = load('trainlist.pkl')

    for k in _trainlist.keys():
        print 'train',k,_trainlist[k]['time']
        idModel.train_ids({k:_trainlist[k]['obs']})
        
    # guardo modelo entrenado en idModel.pkl
    save(idModel,'idModel.pkl')    

    # ---------------------------------------------------------

    # levanto modelo entrenado desde idModel.pkl
    idModel = load('idModel.pkl')
    
    
    # levanto las observaciones de test desde testlist.pkl
    
    _testlist = load('testlist.pkl')

    # inicializo contador de aciertos
    correct = 0
    total = len(_testlist)

    # guardo scores de correctos y de errores
    scores_err = []
    scores_ok = []
    
    # testeo cada paquete de observaciones contra el modelo 
    # y actualizo contador de aciertos
    for k in _testlist.keys():
        value,score = idModel.testObs(_testlist[k]['obs'])        
        print k,'->',value,score
        if check(k,value): 
            correct += 1
            print 'CORRECTO'  
            scores_ok.append(score)
        else: 
            print 'ERROR'
            scores_err.append(score)

    save(scores_ok,'scores_ok.pkl')
    save(scores_err,'scores_err.pkl')   
 
    print 'aciertos: %.2f%%, (%i/%i)'%(100.*correct/total,correct,total)
    print 'train',len(_trainlist),'test',len(_testlist)
    
    
