import numpy as np
from math import pi, exp
from time import time

def fft(X, N):
	'''
	Compute 1D FFT of vector X for N elements
	That is, compute the DFT of 
	x_0, x_s, x_2s x_3s, ... x_(N-1)s
	This is done by breaking up the sum into even and odd parts and recursing down
	This means that there are log(N) recursive steps each summing over N elements
	The end result is a DFT computed in O(N log N)
	@param X - The vector to be computed on
	@param N - The number of elements to be summed over
	'''
	#Base case: Can't sum over anything so just return the element
	if N == 1:
		return X
	
	#Otherwise, separate into even and odd DFTs and recurse
	values = np.copy(X)
	values[0:N/2] = fft(X[0:N:2], N/2)
	values[N/2:N] = fft(X[1:N:2], N/2)
	for k in range(N/2):
		t = values[k]
		values[k] = t + np.exp(-2.0j * pi * k / N) * values[k + N/2]
		values[k + N/2] = t - np.exp(-2.0j * pi * k / N) * values[k + N/2]
	
	return values

def fft2(X, N):
	'''
	Compute the 2D FFT of an NxN matrix X
	This is done by computing the FFT first in the row direction
	Followed by the FFT in the column direction
	@param X - The matrix to be computed on
	@param N - The number of elements to be summed over
	'''
	if N == 1:
		return X
	
	values = np.copy(X)
	for i in range(N):
		values[i, :] = fft(X[i, :], N)
	v2 = np.copy(values)
	for i in range(N):
		v2[:, i] = fft(values[:, i], N)
	
	return v2
	
def main():
		N = 256
		x = np.ones(N, dtype=complex)
		t = time()
		f1 = np.fft.fft(x)
		#print(f1)
		print(time() - t)
		t = time()
		f2 = fft(x, N)
		#print(f2)
		print(time() - t)
		
		print(np.all((f1 - f2) == 0))
		
		x = np.ones([N, N], dtype=complex)
		t = time()
		f1 = np.fft.fft2(x)
		#print(f1)
		print(time() - t)
		t = time()
		f2 = fft2(x, N)
		#print(f2)
		print(time() - t)
		
		print(np.all((f1 - f2) == 0))
		
if __name__== '__main__':
	main()