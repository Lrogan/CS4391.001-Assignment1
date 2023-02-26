import math
import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt    
from Assignment1 import imageComparison

# def erf(x):
#     #constants
#     a1 = 0.254829592
#     a2 = -0.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429
#     p = 0.3275911

#     #A&S formula 7.1.26
#     t = 1.0 / (1.0 + p * abs(x))
#     y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

#     return math.copysign(1, x) * y

# def def_int_gaussian(x, mu, sigma):
#     return 0.5 * erf((x - mu) / (math.sqrt(2) * sigma))

# def gaussian_kernel(kernel_size = 5, sigma = 1, mu = 0, step = 1):
#     end = 0.5 * kernel_size
#     start = -end
#     coeff = []
#     sum = 0
#     x = start
#     last_int = def_int_gaussian(x, mu, sigma)
#     acc = 0
#     while (x < end):
#         x += step
#         new_int = def_int_gaussian(x, mu, sigma)
#         c = new_int - last_int
#         coeff.append(c)
#         sum += c
#         last_int = new_int

#     #normalize
#     sum = 1/sum
#     for i in range(len(coeff)):
#         coeff[i] *= sum
#     return coeff

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()
 
    return kernel_2D

print(repr(gaussian_kernel(15, 2,)))

def convolution(oldimage, kernel):
    #image = Image.fromarray(image, 'RGB')
    image_h = oldimage.shape[0]
    image_w = oldimage.shape[1]
    
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    if(len(oldimage.shape) == 3):
        image_pad = np.pad(oldimage, pad_width=(
            (kernel_h // 2, kernel_h // 2),(kernel_w // 2, 
            kernel_w // 2),(0,0)), mode='constant', 
            constant_values=0).astype(np.float32)
    elif(len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=(
            (kernel_h // 2, kernel_h // 2),(kernel_w // 2, 
            kernel_w // 2)), mode='constant', constant_values=0).astype(np.float32)
    
    
    h = kernel_h // 2
    w = kernel_w // 2
    
    image_conv = np.zeros(image_pad.shape)
    
    for i in range(h, image_pad.shape[0]-h):
        for j in range(w, image_pad.shape[1]-w):
            #sum = 0
            x = image_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w]
            x = x.flatten()*kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w
    
    if(h == 0):
        return image_conv[h:,w:w_end]
    if(w == 0):
        return image_conv[h:h_end,w:]
    return image_conv[h:h_end,w:w_end]

def GaussianBlurImage(sigma):
    #image = imread(image)
    image = cv.imread('../CS4391.001 Assignment 1/sampleImages_Lena/lena_gray.bmp')
    image = np.asarray(image)
    #print(image)
    filter_size = 15
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    
    im_filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        im_filtered[:, :, c] = convolution(image[:, :, c], gaussian_filter)
    return (im_filtered.astype(np.uint8)), image

blur, ori = GaussianBlurImage(2)

imageComparison(blur, "blur", ori, "ori")
