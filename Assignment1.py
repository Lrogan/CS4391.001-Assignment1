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
    plt.subplot(121),plt.imshow(left, cmap="gray"),plt.title(leftLabel)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(right, cmap="gray"),plt.title(rightLabel)
    plt.xticks([]), plt.yticks([])
    plt.show()

def convolution(image, kernel, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))
 
    print("Kernel Shape : {}".format(kernel.shape))
 
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()
 
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()
 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    
    

    return output

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
def cvGaussianFilter(image):
    return cv.GaussianBlur(image, (15,15), 0)

def customGaussianFilter(image, kernelSize):
    
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

    kernel = gaussian_kernel(kernelSize, sigma=math.sqrt(kernelSize))
    return convolution(image, kernel)

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
    def avgSquare(arr, kernel):

        sum = 0
        for i in range(7):
            for j in range(7):
                sum += arr[i][j][0]
        return sum // kernel.shape[0]
    
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

image = getImage('lena_gray.bmp')

Boxing = False
Gausing = False #come back to this later, image on write darker than it should be, looks fine otherwise
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
    imageComparison(cvGaussianBlur, "CV Box Filter", customGaussianBlur, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvGaussianBlur.jpg', cvGaussianBlur)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customGaussianBlur.jpg', customGaussianBlur)
    
if Motion:
    # Creating reference and custom 15x15 Motion blurs
    cvMotionBlur = cvMotionFilter(image)
    customMotionBlur = customMotionFilter(image)
    imageComparison(cvMotionBlur, "CV Box Filter", customMotionBlur, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvMotionBlur.jpg', cvMotionBlur)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customMotionBlur.jpg', customMotionBlur)

if Laplace:
    # Creating reference and custom 3x3 Laplace Sharps
    cvLaplaceSharpening = cvLaplaceFilter(image)
    customLaplaceSharpening = customLaplaceFilter(image)
    imageComparison(cvLaplaceSharpening, "CV Box Filter", customLaplaceSharpening, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvLaplaceSharpening.jpg', cvLaplaceSharpening)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customLaplaceSharpening.jpg', customLaplaceSharpening)
    
if CannyE:
    # Creating reference and custom Edge Detectors
    cvCannyEdge = cvCannyEdgeFilter(image)
    customCannyEdge = customCannyEdgeFilter(image)
    imageComparison(cvCannyEdge, "CV Box Filter", customCannyEdge, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvCannyEdge.jpg', cvCannyEdge)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customCannyEdge.jpg', customCannyEdge)
    

