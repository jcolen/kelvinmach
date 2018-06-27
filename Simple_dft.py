import numpy as np
import matplotlib.pyplot as plt
from math import pi

def f(x, y):
	return 1.0 / (x*x - 2.0*y*y + 5.0j - 5.0)

R = 100
d = 0.1
x = np.arange(-R, R+d, d)
n = len(x)
m = (n - 1) / 2
k = pi / (d * m) * np.arange(-m, m+1, 1)

F = f(x[:, None], x[None, :])

w = np.blackman(n)
W1 = np.reshape(np.repeat(w, n), [n, n])
W2 = np.transpose(W1)
W = W1 * W2

#Have to rotate the grid since dft starts at zero not negative infinity
grid = np.roll(F * W, m, axis=0)
grid = np.roll(grid, m, axis=1)

#Calculate dft
fF = d*d * np.fft.fft2(grid)

#Rotate back to being centered at zero
fF = np.roll(fF, -m, axis=1)
fF = np.roll(fF, -m, axis=0)
fF = np.transpose(fF)

plt.imshow(np.real(fF))
plt.show()