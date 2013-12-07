# To change this template, choose Tools | Templates
# and open the template in the editor.
import numpy
import matplotlib.pyplot as plt

def filterbank(N):
    M=40
    
    fHz = numpy.arange(N)*4000./N
    fmin = 50;
    fmax = 3600;
        
    #fMel = 2595.*numpy.log10(1.+fHz/700.)

    phi_min =  2595.*numpy.log10(1.+fmin/700.)
    phi_max = 2595.*numpy.log10(1.+fmax/700.)
    delta_phi = (phi_max - phi_min)/(M)

    phi_center = numpy.array([phi_min + m*delta_phi for m in range(M+1)])
    fc = 700.*(10.**(phi_center/2595.) - 1.) 
    
    H = numpy.zeros((N,M))
    
    for m in range(1,M):
        fa = fc[m]-fc[m-1]
        fb = fc[m]-fc[m+1]
        for k,f in enumerate(fHz):
            if fc[m-1]<=f and f<fc[m]: H[k,m] = (f-fc[m-1])/fa
            elif fc[m]<=f and f<fc[m+1]: H[k,m] = (f-fc[m+1])/fb
        
        plt.plot(fHz, H[:,m])
    
    #print numpy.shape(H)
    #print numpy.shape(f)
    print fc
    plt.show()
    
    
filterbank(160)