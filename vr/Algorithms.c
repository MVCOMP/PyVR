/*
 
 Método de compilación para python:
 
gcc-4.2 -c Algorithms.c && gcc-4.2 -shared -o libvr.so Algorithms.o
gcc -c Algorithms.c && gcc -shared -o libvr.so Algorithms.o

 */

#include "Algorithms.h"

#include <math.h>
#include <limits.h>
#include <float.h>
#include <memory.h>

#if _DEBUG
#include <stdio.h>
#include <stdlib.h>
#endif


// Algoritmo modificado...
float Algorithms_BhattacharyaDistance(float * ua, float * ub, float * sa, float * sb, unsigned int Q)
{
    unsigned int i;
    float A;
    double d = 0.0;
    for (i=0; i < Q; ++i ) {
        A = (sa[i] + sb[i])/2.0;
        d += pow((double)(ua[i]-ub[i]),2.0)/A;
        d += 2.0 * log((double)A);
        d -= log((double)(sa[i]*sb[i]));
    }
    // No es necesario restar una constante ni divdir a los efectos de comparar
    //d -= 1.3862943612 * MFCC_SIZE;
    //d /= 4.0;
    return (float)d;
}



void Algorithms_ButterworthFilter(float *x, float *y, unsigned int n)
{
    int i,j;
    
    for (i=0; i<BUTT_ORD; i++)
    {
        y[i] = BUTT_B[0]*x[i];
        
        for (j=1; j<=i; j++)
            y[i] = y[i] + BUTT_B[j] * x[i-j] - BUTT_A[j] * y[i-j];
    }
    
    for (i=BUTT_ORD; i<n; i++)
    {
        y[i] = BUTT_B[0]*x[i];
        
        for (j=1; j<=BUTT_ORD; j++)
            y[i] = y[i] + BUTT_B[j] * x[i-j] - BUTT_A[j] * y[i-j];
    }
}

void dft(float * in, float * out)
{
    int k,t;
    float sumreal;
    float sumimag;

    // calculo unicamente para el intervalo que se va a usar en el MFCC
    //for (k = MFCC_FILTER_KI[0]; k < MFCC_FILTER_KF[MFCC_SIZE-1]; k++)
    for (k = 0; k < WINDOW_SIZE; k++)
    {
        sumreal = 0.0;
        sumimag = 0.0;
        for (t=0;t<WINDOW_SIZE;t++)
        {
            sumreal += in[t]*cos(2.0*PI*t*k/WINDOW_SIZE);
            sumimag += -in[t]*sin(2.0*PI*t*k/WINDOW_SIZE);
        }
        out[k] = (float)sqrt((double)(sumreal*sumreal + sumimag*sumimag));
    }
}

// mfcc_coeficients -> vector de 20 coeficientes MFCC
void mfcc(float * window, float * mfcc_coefs )
{
    unsigned int m,k;
    
    float dft_module[WINDOW_SIZE];
    float h, Ar;

    dft(window, dft_module);
    
    memset(mfcc_coefs, 0, sizeof(float) * MFCC_SIZE);
    
    for ( m = 0; m < MFCC_SIZE; m++){
        Ar = 0.0f;
        for(k=0; k<WINDOW_SIZE/2; k++){
            if(FREQS[k] < FILTERBANK[m+1] && FREQS[k] >= FILTERBANK[m])
            {
                h = (FREQS[k]-FILTERBANK[m])/(FILTERBANK[m+1]-FILTERBANK[m]);
                mfcc_coefs[m] += h * dft_module[k];
                Ar += h;
            }
            else if(FREQS[k] < FILTERBANK[m+2] && FREQS[k] >= FILTERBANK[m+1])
            {
                h = (FREQS[k]-FILTERBANK[m+2])/(FILTERBANK[m+1]-FILTERBANK[m+2]);
                mfcc_coefs[m] += h * dft_module[k];
                Ar += h;
            }
        }
        mfcc_coefs[m] = (float)log((double)(mfcc_coefs[m]/Ar));
    }
    dct(mfcc_coefs);
}

void dct(float * mfcc_coefs)
{
	int l,m;
    double c;
    
    float coeficients[MFCC_SIZE];
    memset(coeficients,0,sizeof(float) * MFCC_SIZE);
    
	for(l = 0; l < MFCC_SIZE; l++)
	{
		c = (double)l*PI/MFCC_SIZE;
        
		for (m = 0; m < MFCC_SIZE; m++)
		{
			coeficients[l] += mfcc_coefs[m] * (float)cos(c*(m+0.5));
		}
	}
    memcpy(mfcc_coefs, coeficients, sizeof(float) * MFCC_SIZE);
}

void Algorithms_FrameProcess(float * x, float * mfcc_coefs)
{
    unsigned int i;
    float zcr = 0.0;
    int sign = -1;

    // Hamming
    float xHamming[WINDOW_SIZE];
    float xPE[WINDOW_SIZE];

    memset(xHamming,0,sizeof(float) * WINDOW_SIZE);
    
    // Preenfasis
    memset(xPE,0,sizeof(float) * WINDOW_SIZE);
            
    x[0] /= 32768.0;
    xHamming[0] = x[0] * HAMMING_WINDOW[0];
    xPE[0] = xHamming[0];
    
    
    for (i=1; i<WINDOW_SIZE; ++i)
    {
        // Normalizacion
        x[i] /= 32768.0;
        
        if (sign==-1 && x[i]>=0){
            zcr ++;
            sign = 1;
        }
        else if (sign==1 && x[i]<0){
            zcr ++;
            sign = -1;
        }
        
        // Hamming
        xHamming[i] = x[i] * HAMMING_WINDOW[i];
        // Preenfasis
        xPE[i] = xHamming[i] - 0.95 * xHamming[i-1];
    }
    zcr /= WINDOW_SIZE;
    
    // mfcc
    mfcc(xPE,mfcc_coefs);
    mfcc_coefs[0] = zcr;
    
}


