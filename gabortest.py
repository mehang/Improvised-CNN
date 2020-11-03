from skimage.filters import gabor_kernel
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np

import math


def get_gabor_filters(inchannels, outchannels, kernel_size=(3, 3)):
    delta = 1e-4
    freqs = (math.pi / 2) * math.sqrt(2) ** (-np.random.randint(0, 5, (outchannels, inchannels)))
    thetas = (math.pi / 8) * np.random.randint(0, 8, (outchannels, inchannels))
    sigmas = math.pi / freqs
    psis = math.pi * np.random.rand(outchannels, inchannels)
    x0, y0 = np.ceil(np.array(kernel_size) / 2)

    y, x = np.meshgrid(
        np.linspace(-x0 + 1, x0 + 0, kernel_size[0]),
        np.linspace(-y0 + 1, y0 + 0, kernel_size[1]),
    )
    filterbank = []

    for i in range(outchannels):
        for j in range(inchannels):
            freq = freqs[i][j]
            theta = thetas[i][j]
            sigma = sigmas[i][j]
            psi = psis[i][j]

            rotx = x * np.cos(theta) + y * np.sin(theta)
            roty = -x * np.sin(theta) + y * np.cos(theta)

            g = np.exp(
                -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + delta) ** 2)
            )
            g = g * np.cos(freq * rotx + psi)
            g = g / (2 * math.pi * sigma ** 2)
            g = gabor_kernel(frequency=freq, bandwidth=sigma, theta=theta, n_stds=0).real
            filterbank.append(g)
    return filterbank


filterbank = get_gabor_filters(3, 64, (3, 3))

fig = plt.subplots(12, 16, figsize=(22, 22))
for i, gf in enumerate(filterbank):
    plt.subplot(12, 16, i + 1)
    plt.imshow(gf, cmap='gray')
    plt.axis('off')