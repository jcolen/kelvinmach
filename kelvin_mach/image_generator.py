import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv
from generate_wake_grid import generate_wake_grid

#from mayavi import mlab

R = 64
d = 0.125

if len(argv) > 1:
	phi, n = generate_wake_grid(float(argv[1]), float(argv[2]), float(argv[3]), float(argv[4]), R, d)
		
	fig = plt.figure()
	plt.imshow(np.real(phi))
	#fig = plt.figure()
	#plt.imshow(np.real(n))
	plt.show()
	
	'''
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, np.real(phi))
	fig = plt.figure();
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, np.real(n))
	'''
	'''
	x = np.arange(-R, R+d, d)
	y = np.arange(-R, R+d, d)
	X, Y = np.meshgrid(x, y)
	mlab.figure(bgcolor=(1,1,1))
	mlab.surf(x, y, np.real(phi))
	mlab.show()
	'''
else:
	#for M in np.arange(1.0, 5.0, 0.5):
	#for M in [1.4, 2.5, 4.0]:
	for M in [4.0]:
		#for l in [0.01, 0.05, 0.1, 0.2, 1]:
		for l in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
			phi, n = generate_wake_grid(M, 1.0, 1.0, l, R, d)
			filename = 'wake_M%f_lambda%f' % (M, l)
			fig = plt.figure()
			plt.imshow(np.real(phi))
			print(filename)
			fig.savefig('C:\Users\Jonathan\Documents\School\Kolomeisky\\phi_ext_patterns\%s.png' % filename)
			plt.close(fig)