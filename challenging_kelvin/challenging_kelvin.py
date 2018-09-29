import numpy as np
from math import pi
from sys import argv
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'

def phi_in(kx, ky, l):
    k = np.sqrt(kx*kx + ky*ky)
    return k / ((kx + l*1.0j) * (kx + l*1.0j) - k)
    
def phi_in2(kx, ky, l):
    k = np.sqrt(kx*kx + ky*ky)
    return kx*kx / ((kx + l*1.0j) * (kx + l*1.0j) - k)
    
def calculate_wake(l, R, d, opt=0):
    #Generate grid of points for fast fourier transform
    k = np.arange(-R, R+d, d)
    n = len(k)

    m = (n - 1) / 2
    x = pi / (d * m) * np.arange(-m, m+1, 1)
    
    #Calculate the function values at each point in the grid
    if opt == 0:
        phi_induced = phi_in(k[:, None], k[None, :], l)
    else:
        phi_induced = phi_in2(k[:, None], k[None, :], l)
    
    #Have to rotate the grid because dft starts at zero not negative infinity
    phi_grid = np.roll(phi_induced, -m, axis=0)
    phi_grid = np.roll(phi_grid, -m, axis=1)
    
    #Calculate dft
    fPhi = d*d * np.fft.ifft2(phi_grid)

    #Rotate back to being centered at zero
    fPhi = np.roll(fPhi, m, axis=1)
    fPhi = np.roll(fPhi, m, axis=0)
    fPhi = np.transpose(fPhi)    #Have to transpose result to make it look pretty
    
    return x, fPhi
    
R = 64
d = 0.0625

if __name__ == '__main__':
    if len(argv) > 1:
        opt = int(argv[3]) if len(argv) > 3 else 0
        r = int(argv[2]) if len(argv) > 2 else R
        x, phi = calculate_wake(float(argv[1]), r, d, opt)
        xs = x[-1]
        fig = plt.figure()
        plt.imshow(np.real(phi), extent=[-xs, xs, -xs, xs])
        plt.plot(x, np.sqrt(x**2.0 / 8.), color='blue')
        plt.plot(x, -np.sqrt(x**2.0 / 8.), color='blue')
        plt.xlim([-xs, 0])
        plt.ylim([-xs/2, xs/2])
        plt.colorbar()
        plt.show()
        print(np.min(phi[:]), np.max(phi[:]))
    else:
        l = 0.0
        step = 0.0001
        for i in range(100):
            l += step
            x, phi = calculate_wake(l, R, d)
            half = phi.shape[0] / 2
            phi = phi[half/2:-half/2, half:]
            filename = 'lambda_%f' % l
            fig = plt.figure()
            plt.imshow(np.real(phi))
            print(filename)
            #fig.savefig('C:\Users\Jonathan\Documents\School\Kolomeisky\\challenging_kelvin\%s.png' % filename)
            plt.close(fig)
