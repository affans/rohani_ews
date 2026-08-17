import math
import numpy as np
from libc.stdlib cimport malloc, free


### Kolmogorov complexity ###

#% a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976),
#% h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
#% and h(n) is a normalised measure of complexity. This function returns h(n) 


def normedKolmogorovComplexity(s):
	n = len(s)
	c = 1
	l = 1
	b = n/math.log(n,2);	
	i = 0
	k = 1
	k_max = 1
	stop = 0

	while stop==0:
		if s[i+k-1] != s[l+k-1]:
			if k>k_max:
				k_max=k
			i=i+1
			if i==l:
				c=c+1
				l=l+k_max
				if l+1>n:
					stop=1
				else:
					i=0
					k=1
					k_max=1
			else:
				k=1
		else:
			k=k+1
			if l+k>n:
				c=c+1
				stop=1			
	return(c/b)
	
def CnormedKolmogorovComplexity(s):
	cdef int n = len(s)
	cdef int p = 10

	cdef int c = 1
	cdef int l = 1
	cdef int i = 0
	cdef int k = 1
	cdef int k_max = 1
	cdef int stop = 0
	
	cdef int *cs = <int *>malloc(n * sizeof(int))
	try:
		for j in range(n): cs[j] = s[j]
		while stop==0:
			if cs[i+k-1] != cs[l+k-1]:
				if k>k_max:
					k_max=k
				i=i+1
				if i==l:
					c=c+1
					l=l+k_max
					if l+1>n:
						stop=1
					else:
						i=0
						k=1
						k_max=1
				else:
					k=1
			else:
				k=k+1
				if l+k>n:
					c=c+1
					stop=1
	finally:
		free(cs)
				
	b = n/math.log(n,2);			
	return(c/b)		

def MovingKC(data, mean, window):
	n = len(data)
	bindata = binaryTimeseries(data,mean)
	kc = np.array([normedKolmogorovComplexity(bindata[i-window:i]) for i in range(2*window, n)])
	na = np.zeros(2*window)
	na.fill(float('NaN'))
	kc = np.concatenate([na,kc])
	return(kc) 

def CMovingKC(data, mean, window):
	n = len(data)
	bindata = binaryTimeseries(data,mean)
	kc = np.array([CnormedKolmogorovComplexity(bindata[i-window:i]) for i in range(2*window, n)])
	na = np.zeros(2*window)
	na.fill(float('NaN'))
	kc = np.concatenate([na,kc])
	return(kc) 		
	
#for detrending:	
from scipy import signal
	
def CMovingKC_detrend(data, window):
	n = len(data)	
	def KCbinaryTimeseriesDetrend(data,window):
		zeros = np.zeros(window)
		detrend = signal.detrend(data, type='linear')
		binary = binaryTimeseries(detrend,zeros)
		return(CnormedKolmogorovComplexity(binary))
	kc = np.array( [KCbinaryTimeseriesDetrend(data[i-window:i],window) for i in range(2*window, n)])
	na = np.zeros(2*window)
	na.fill(float('NaN'))
	kc = np.concatenate([na,kc])
	return(kc) 		
	
###  ###

#% Convert a timeseries, a, to a binary timeseries, c. If a[i] >= b[i] then c[i] = 1, otherwise c[i] = 0. Taken from Rohani and Miramontes, 1995. Takes as an argument two numpy arrays

def binaryTimeseries(a,b):

	def switch(x):
		if(x>=0):
			return(1)
		else:
			return(0)

	return([switch(x) for x in a-b])
	
	
	
	


