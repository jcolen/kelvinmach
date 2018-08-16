import numpy as np
from math import pi

def phi_in(qx, qy, M, g, s, l):
	return 1.0 / ((M * qx + l * 1.0j) * (M * qx + l * 1.0j) - g * np.sqrt(qx*qx + qy*qy) - s*s * (qx*qx + qy*qy))

def n_in(qx, qy, M, g, s, l):
	return np.sqrt(qx * qx + qy * qy) / ((M * qx + l * 1.0j) * (M * qx + l * 1.0j) - g * np.sqrt(qx*qx + qy*qy) - s*s * (qx*qx + qy*qy))
	
def generate_wake_grid(M, g, s, l, R, d):
	#Generate grid of points for fast fourier transform
	x = np.arange(-R+d, R+d, d)
	n = len(x)

	m = (n - 1) / 2
	k = pi / (d * m) * np.arange(-m, m+1, 1)
	
	#Calculate the function values at each point in the grid
	phi_induced = phi_in(x[:, None], x[None, :], M, g, s, l)
	n_induced = n_in(x[:, None], x[None, :], M, g, s, l)
	
	#Generate windowing function to handle emphasizing nearby points in FFT
	#Using blackman window
	#w = np.blackman(n)
	w = np.hamming(n)
	W1 = np.reshape(np.repeat(w, n), [n, n])
	W2 = np.transpose(W1)
	W = W1 * W2
	
	#Have to rotate the grid because dft starts at zero not negative infinity
	phi_grid = np.roll(phi_induced * W, -m, axis=0)
	phi_grid = np.roll(phi_grid, -m, axis=1)
	n_grid = np.roll(n_induced * W, -m, axis=0)
	n_grid = np.roll(n_grid, -m, axis=1)
	
	#Calculate dft
	fPhi = d*d * np.fft.fft2(phi_grid)
	fN = d*d * np.fft.fft2(n_grid)
	
	#Rotate back to being centered at zero
	fPhi = np.roll(fPhi, m, axis=1)
	fPhi = np.roll(fPhi, m, axis=0)
	fPhi = np.transpose(fPhi)	#Have to transpose result to make it look pretty
	
	fN = np.roll(fN, m, axis=1)
	fN = np.roll(fN, m, axis=0)
	fN = np.transpose(fN)
	
	return fPhi, fN