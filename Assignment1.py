import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hide import kernel #this is to get the copy and pasted 15x15 Gaussian kernal array out of the main body of code

# -----------------------------------Everything Uses These------------------------------------------------
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


# -----------------------------------Box Filters------------------------------------------------
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

# -----------------------------------Gaussian Filter------------------------------------------------
def cvGaussianFilter(image):
    return cv.GaussianBlur

def customGaussianFilter(image):
    # Determine the avg of the current kernel
    def weightedAvgSquare(arr):
        weightedArr = np.dot(arr, kernel)
        sum = 0
        for i in range(15):
            for j in range(15):
                sum += weightedArr[i][j][0]
        return sum // 49
    
    # Setup the Kernel, Kernel Weights, blur, and indicies
    currSquare = []
    currSquare_row = []

    blur_row = []
    blur = []

    rows = len(image)
    cols = len(image[0])

    row, col = 0, 0

    while row <= rows - 15:
        while col <= cols - 15:

            for i in range(row, row + 15):

                for j in range(col, col + 15):

                    currSquare_row.append(image[i][j])

                currSquare.append(np.asarray(currSquare_row))
                currSquare_row = []

            avg = weightedAvgSquare(currSquare)
            blur_row.append(np.asarray([avg, avg, avg])) #idc if its color, its becoming grey
            currSquare = []

            col += 1
        
        blur.append(np.asarray(blur_row))
        blur_row = []

        row += 1
        col = 0
    
    return np.asarray(blur)

image = getImage('lena_gray.bmp')

Boxing = False
Gausing = True
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
    # Creating reference and custom 7x7 box blurs
    cvGaussianBlur = cvGaussianFilter(image)
    customGaussianBlur = customGaussianFilter(image)
    imageComparison(cvGaussianBlur, "CV Box Filter", customGaussianBlur, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvGaussianBlur.jpg', cvGaussianBlur)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customGaussianBlur.jpg', customGaussianBlur)
    
if Motion:
    # Creating reference and custom 7x7 box blurs
    cvMotionBlur = cvMotionFilter(image)
    customMotionBlur = customMotionFilter(image)
    imageComparison(cvMotionBlur, "CV Box Filter", customMotionBlur, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvMotionBlur.jpg', cvMotionBlur)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customMotionBlur.jpg', customMotionBlur)

if Laplace:
    # Creating reference and custom 7x7 box blurs
    cvLaplaceSharpening = cvLaplaceFilter(image)
    customLaplaceSharpening = customLaplaceFilter(image)
    imageComparison(cvLaplaceSharpening, "CV Box Filter", customLaplaceSharpening, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvLaplaceSharpening.jpg', cvLaplaceSharpening)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customLaplaceSharpening.jpg', customLaplaceSharpening)
    
if CannyE:
    # Creating reference and custom 7x7 box blurs
    cvCannyEdge = cvCannyEdgeFilter(image)
    customCannyEdge = customCannyEdgeFilter(image)
    imageComparison(cvCannyEdge, "CV Box Filter", customCannyEdge, "Custom Box Filter")
    # Saving blured images into separate folders
    cv.imwrite('../CS4391.001 Assignment 1/OpenCV_Output/cvCannyEdge.jpg', cvCannyEdge)
    cv.imwrite('../CS4391.001 Assignment 1/Custom_Output/customCannyEdge.jpg', customCannyEdge)
    

