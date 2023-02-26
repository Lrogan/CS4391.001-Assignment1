import math
import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# -----------------------------------Everything Uses These------------------------------------------------ #
# Its expecteding to find a folder named sampleImages_Lena that contains all the images
imagesRootPath = '../CS4391.001 Assignment 1/sampleImages_Lena/'

# Gets the image according to the filename, must be in the folder hardcoded into the imagesRootPath var
def getImage(filename):
    return cv.imread(imagesRootPath + filename)

# shows two images with custom labels
def imageComparison(left, leftLabel, right, rightLabel):
    plt.subplot(121),plt.imshow(left),plt.title(leftLabel)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(right),plt.title(rightLabel)
    plt.xticks([]), plt.yticks([])
    plt.show()

# ----------------------------------- Box Filters ------------------------------------------------ #
# Creates OpenCV Example 7x7 Box Blur
def cvBoxFilter(image):
    return cv.blur(image, (7,7))

# Custom 7x7 Box Blur without OpenCV
def customBoxFilter(image):

    # Determine the avg of the current kernel
    def avgSquare(arr):
        sum = 0
        for i in range(7):
            for j in range(7):
                sum += arr[i][j][0]
        return sum // 49
    
    # Setup the Kernel, blur, and indicies
    currSquare = []
    currSquare_row = []

    blur_row = []
    blur = []

    rows = len(image)
    cols = len(image[0])

    row, col = 0, 0

    while row <= rows - 7:
        while col <= cols - 7:

            for i in range(row, row + 7):

                for j in range(col, col + 7):

                    currSquare_row.append(image[i][j])

                currSquare.append(np.asarray(currSquare_row))
                currSquare_row = []

            avg = avgSquare(currSquare)
            blur_row.append(np.asarray([avg, avg, avg]))
            currSquare = []

            col += 1
        
        blur.append(np.asarray(blur_row))
        blur_row = []

        row += 1
        col = 0
    
    return np.asarray(blur)

# ----------------------------------- Gaussian Filters------------------------------------------------ #
def cvGaussianFilter(image, size=15):
    return cv.GaussianBlur(image, (size,size), 0)

def customGaussianFilter(image, kernelSize):  
    
    def gaussian_kernel(size):
        if size == 15:
            kernel_1D = np.array([0.00048872837522002, 0.002403157286908872, 0.009246250740395456,
                        0.027839605612666265, 0.06560233156931679, 0.12099884565428047,
                        0.1746973469158936, 0.19744746769063704, 0.1746973469158936,
                        0.12099884565428047, 0.06560233156931679, 0.027839605612666265,
                        0.009246250740395456, 0.002403157286908872, 0.00048872837522002 ]) # taken from the kernel generator
        else:
            kernel_1D = np.array([0.31946576033846985, 0.3610684793230603, 0.31946576033846985]) # taken from the kernel generator

        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    
        kernel_2D *= 1.0 / kernel_2D.max()
    
        return kernel_2D

    kernel = gaussian_kernel(kernelSize)

    def GaussianSquare(arr, kernel):
        sum = 0
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[0]):
                sum += arr[i][j][0]*kernel[i][j]
        return sum // (kernel.shape[0] ** 2 - 200)
    
    # Setup the Kernel, blur, and indicies
    currSquare = []
    currSquare_row = []

    blur_row = []
    blur = []

    rows = len(image)
    cols = len(image[0])

    row, col = 0, 0

    while row <= rows - kernelSize:
        while col <= cols - kernelSize:

            for i in range(row, row + kernelSize):

                for j in range(col, col + kernelSize):

                    currSquare_row.append(image[i][j])

                currSquare.append(np.asarray(currSquare_row))
                currSquare_row = []

            avg = GaussianSquare(currSquare, kernel)
            blur_row.append(np.asarray([avg, avg, avg]))
            currSquare = []

            col += 1
        
        blur.append(np.asarray(blur_row))
        blur_row = []

        row += 1
        col = 0
    
    return np.asarray(blur)
    # return convolution(image, kernel)

# ----------------------------------- Motion Blur ------------------------------------------------ #
# Generate the Motion Kernel
def generateMotionKernel(size):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    return kernel / size

# Creates OpenCV Example 15x15 Motion Blur
def cvMotionFilter(image, kernel):
    return cv.filter2D(image, -1, kernel)

# Custom 15x15 Motion Blur without OpenCV
def customMotionFilter(image, size):
    kernel = generateMotionKernel(size)
    # Determine the avg of the current kernel
    def motionSquare(arr, kernel):
        sum = 0
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[0]):
                sum += arr[i][j][0]*kernel[i][j]
        return sum
    
    # Setup the Kernel, blur, and indicies
    currSquare = []
    currSquare_row = []

    blur_row = []
    blur = []

    rows = len(image)
    cols = len(image[0])

    row, col = 0, 0

    while row <= rows - size:
        while col <= cols - size:

            for i in range(row, row + size):

                for j in range(col, col + size):

                    currSquare_row.append(image[i][j])

                currSquare.append(np.asarray(currSquare_row))
                currSquare_row = []

            avg = motionSquare(currSquare, kernel)
            blur_row.append(np.asarray([avg, avg, avg]))
            currSquare = []

            col += 1
        
        blur.append(np.asarray(blur_row))
        blur_row = []

        row += 1
        col = 0
    
    return np.asarray(blur)

# ----------------------------------- Laplace Sharpening ------------------------------------------------ #
def generateLaplaceKernel(size):
    zerosAnd2 = np.zeros((size, size))
    zerosAnd2[1][1] = 2
    OnesOver9 = np.ones((size,size))/9
    kernel = zerosAnd2 - OnesOver9
    return kernel

def cvLaplaceFilter(image):
    return image

def customLaplaceFilter(image, size):
    kernel = generateLaplaceKernel(size)
    # Determine the avg of the current kernel
    def laplaceSquare(arr, kernel):
        sum = 0
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[0]):
                pp = arr[i][j][0]*kernel[i][j] #pixel product
                sum += pp
        return sum
    
    # Setup the Kernel, blur, and indicies
    currSquare = []
    currSquare_row = []

    blur_row = []
    blur = []

    rows = len(image)
    cols = len(image[0])

    row, col = 0, 0

    while row <= rows - size:
        while col <= cols - size:

            for i in range(row, row + size):

                for j in range(col, col + size):

                    currSquare_row.append(image[i][j])

                currSquare.append(np.asarray(currSquare_row))
                currSquare_row = []

            avg = laplaceSquare(currSquare, kernel)
            blur_row.append(np.asarray([avg, avg, avg]))
            currSquare = []

            col += 1
        
        blur.append(np.asarray(blur_row))
        blur_row = []

        row += 1
        col = 0
    
    return np.asarray(blur)

# ----------------------------------- Canny Edge Detection ------------------------------------------------ #



# ----------------------------------- Driver Code ------------------------------------------------ #
image = getImage('lena_gray.bmp')

Boxing = False
Gausing = False
Motion = False
Laplace = False
CannyE = False

if Boxing:
    # Creating reference and custom 7x7 box blurs
    cvBoxBlur = cvBoxFilter(image)
    customBoxBlur = customBoxFilter(image)
    imageComparison(cvBoxBlur, "CV Box Filter", customBoxBlur, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvBoxBlur.jpg', cvBoxBlur)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customBoxBlur.jpg', customBoxBlur)

if Gausing:
    # Creating reference and custom 15x15 Gaussian blurs
    cvGaussianBlur = cvGaussianFilter(image)
    customGaussianBlur = customGaussianFilter(image, 15)
    imageComparison(cvGaussianBlur, "CV Gauss Filter", customGaussianBlur, "Custom Gauss Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvGaussianBlur.jpg', cvGaussianBlur)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customGaussianBlur.jpg', customGaussianBlur)
    
if Motion:
    # Creating reference and custom 15x15 Motion blurs
    cvMotionBlur = cvMotionFilter(image, generateMotionKernel(15))
    customMotionBlur = customMotionFilter(image, 15)
    # imageComparison(cvMotionBlur, "CV Box Filter", customMotionBlur, "Custom Box Filter") # for some reason this doesn't work, maybe because of the grey scale I have but meh
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvMotionBlur.jpg', cvMotionBlur)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customMotionBlur.jpg', customMotionBlur)

if Laplace:
    # Creating reference and custom 3x3 Laplace Sharps
    cvLaplaceSharpening = cvLaplaceFilter(image)
    customLaplaceSharpening = customLaplaceFilter(image, 3)
    imageComparison(cvLaplaceSharpening, "CV Laplace Filter", customLaplaceSharpening, "Custom Laplace Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvLaplaceSharpening.jpg', cvLaplaceSharpening)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customLaplaceSharpening.jpg', customLaplaceSharpening)
    
if CannyE:
    # Creating reference and custom Edge Detectors
    cvCannyEdge = cvCannyEdgeFilter(image)
    customCannyEdge = customCannyEdgeFilter(image)
    imageComparison(cvCannyEdge, "CV Canny Edge Filter", customCannyEdge, "Custom Canny Edge Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvCannyEdge.jpg', cvCannyEdge)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customCannyEdge.jpg', customCannyEdge)
    

