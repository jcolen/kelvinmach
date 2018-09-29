import numpy as np
from math import pi, exp
import scipy.special as scsp
import matplotlib.pyplot as plt

def charge_away_plane(qx, qy, a=5.0):
	return np.exp(-1.0 * np.sqrt(qx*qx + qy*qy) * a)

def gaussian_in_plane(qx, qy, a=2.0):
	return np.exp(-(qx*qx + qy*qy)*a*a)
	
def charged_rectangle(qx, qy, a=2.0, b=0.0):
	return np.real(np.sin(qx*a + 0.001j) / (qx*a + 0.001j)) * \
		np.real(np.sin(qy*b + 0.001j) / (qy*b + 0.001j))

def charged_disc(qx, qy, a=2.0):
	q = np.sqrt(qx * qx + qy * qy)
	return np.real(scsp.jn(1, q * a + 0.001j) * 2 / (q * a + 0.001j))

def delta(qx, qy, a=2.0):
	return 1.0

def phi_in(qx, qy, M, g, s, l, next, a=2.0):
	return next(qx, qy, a) / ((M * qx + l * 1.0j) * (M * qx + l * 1.0j) - g * np.sqrt(qx*qx + qy*qy) - s*s * (qx*qx + qy*qy))
	
def generate_wake_grid(M, g, s, l, R, d, next, a=2.0):
	#Generate grid of points for fast fourier transform
	x = np.arange(-R+d, R+d, d)
	n = len(x)

	m = (n - 1) / 2
	k = pi / (d * m) * np.arange(-m, m+1, 1)
	
	#Calculate the function values at each point in the grid
	phi_induced = phi_in(x[:, None], x[None, :], M, g, s, l, next, a)
	
	#Generate windowing function to handle emphasizing nearby points in FFT
	#Also prevents issues created by long tails in the q-space representation
	#Using blackman window
	#w = np.blackman(n)
	w = np.hamming(n)
	W1 = np.reshape(np.repeat(w, n), [n, n])
	W2 = np.transpose(W1)
	W = W1 * W2
	
	#Have to rotate the grid because dft starts at zero not negative infinity
	phi_grid = np.roll(phi_induced * W, -m, axis=0)
	phi_grid = np.roll(phi_grid, -m, axis=1)
	
	#Calculate dft
	#Using numpy's FFT2 instead of the one I wrote because it runs fast
	#TODO write a FFT2 program in C and see how fast it can be made
	fPhi = d*d * np.fft.fft2(phi_grid)
	
	#Rotate back to being centered at zero
	fPhi = np.roll(fPhi, m, axis=1)
	fPhi = np.roll(fPhi, m, axis=0)
	fPhi = np.transpose(fPhi)	#Have to transpose result to make it look pretty
	
	return fPhi

def main():
	M = 1.0 #4.0 
	l = 0.09
	R = 128
	d = 0.125
	a = 2.0
	s = 1.0
	#phi = generate_wake_grid(M, 1.0, 1.0, l, R, d, delta)
	
	#phi = generate_wake_grid(M, 1.0, 1.0, l, R, d, charge_away_plane)
	#filename = 'wake_M%f_lambda%f_%s' % (M, l, 'charge_away_plane')
	phi = generate_wake_grid(M, 1.0, 0.0, l, R, d, gaussian_in_plane)
#filename = 'wake_M%f_lambda%f_%s' % (M, l, 'gaussian_in_plane')
	#phi = generate_wake_grid(M, 1.0, 1.0, l, R, d, charged_rectangle)
	#filename = 'wake_M%f_lambda%f_%s' % (M, l, 'charged_rectangle')
	#phi = generate_wake_grid(M, 1.0, 1.0, l, R, d, charged_disc)
	#filename = 'wake_M%f_lambda%f_%s' % (M, l, 'charged_disc')
	'''
	for i in range(20):
		a = 0.2 + 0.1 * i
		phi = generate_wake_grid(M, 1.0, 0.0, l, R, d, gaussian_in_plane, a=a)
		filename = 'wake_M%f_lambda%f_%s_a%f' % (M, l, 'gaussian_in_plane', a)
		print(filename)
		fig = plt.figure()
		plt.imshow(np.real(phi))
		fig.savefig('C:\Users\Jonathan\Documents\School\Kolomeisky\\phi_ext_patterns\%s.png' % filename)
		plt.close(fig)
	'''
	'''
	for a in [0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
		phi = generate_wake_grid(M, 1.0, s, l, R, d, charged_rectangle, a)
		if s == 0:
			filename = 'wake_M%f_lambda%f_a%f_%s_Kelvin' % (M, l, a, 'charged_rectangleb=0')
		else:
			filename = 'wake_M%f_lambda%f_a%f_%s' % (M, l, a, 'charged_rectangleb=0')
		fig = plt.figure()
		plt.imshow(np.real(phi))
		fig.savefig('C:\Users\Jonathan\Documents\School\Kolomeisky\\phi_ext_patterns\\charged_rectangle_b=0\\%s.png' % filename)
		plt.close(fig)
	'''	
	fig = plt.figure()
	plt.imshow(np.real(phi))
	#fig.savefig('C:\Users\Jonathan\Documents\School\Kolomeisky\\phi_ext_patterns\%s.png' % filename)
	plt.colorbar()
	plt.show()
	
	
if __name__ == '__main__':
	main()
