#!/usr/bin/env python
# coding: utf-8
'''
 FFT, p(k) 

'''

import numpy as np

# Set seeds
np.random.seed(10)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt


swe_data = np.log10(np.load("../data/Zeldotest.np.npy"))

swe_data = swe_data.reshape(1000,64*64)


def get_potential_gradients(den_real):
    """Starting from a density field in 2D, get the potential gradients i.e.
    returns the two components of grad (grad^-2 den_real)"""
    den = np.fft.fft2(den_real)
    freqs = np.fft.fftfreq(den.shape[0])
    del_sq_operator = -(freqs[:,np.newaxis]**2+freqs[np.newaxis,:]**2)

    grad_x_operator = -1.j*np.fft.fftfreq(den.shape[0])[:,np.newaxis]
    grad_y_operator = -1.j*np.fft.fftfreq(den.shape[0])[np.newaxis,:]

    phi = den/del_sq_operator
    removeNaN(phi)

    grad_phi_x = grad_x_operator*phi
    grad_phi_y = grad_y_operator*phi

    grad_phi_x_real = np.fft.ifft2(grad_phi_x).real
    grad_phi_y_real = np.fft.ifft2(grad_phi_y).real

    return grad_phi_x_real, grad_phi_y_real

image = swe_data[1]
from scipy import fftpack
# Take the fourier transform of the image.
F1 = fftpack.fft2(image)

# Now shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
F2 = fftpack.fftshift( F1 )

# Calculate a 2D power spectrum
psd2D = np.abs( F2 )**2

# Calculate the azimuthally averaged 1D power spectrum
psd1D = radialProfile.azimuthalAverage(psd2D)

# Now plot up both
plt.figure(1)
plt.clf()
plt.imshow( np.log10( image ), cmap=py.cm.Greys)

plt.figure(2)
plt.clf()
plt.imshow( np.log10( psf2D ))

plt.figure(3)
plt.clf()
plt.semilogy( psf1D )
plt.xlabel('Spatial Frequency')
plt.ylabel('Power Spectrum')

plt.show()