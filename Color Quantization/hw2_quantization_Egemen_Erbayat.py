import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

label = 1

def integralSum(array):
    h, w = array.shape[0], array.shape[1]
    output = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if j == 0 and i == 0:
                output[i][j] = array[i][j]
            elif i == 0 and j != 0:
                output[i][j] = output[i][j-1] + array[i][j]
            elif i != 0 and j == 0:
                    output[i][j] = output[i-1][j] + array[i][j]
            else:
                    output[i][j] = output[i-1][j] + output[i][j-1] - output[i-1][j-1] + array[i][j]
    return output

def calculateMean(integralSum,windowSize,x,y,h,w):
    d = windowSize//2
    xSmall = max(x - d,0)
    ySmall = max(y - d,0)
    xLarge = min(x + d -1,w-1)
    yLarge = min(y + d - 1,h-1)
    sum = integralSum[ySmall][xSmall] + integralSum[yLarge][xLarge] - integralSum[yLarge][xSmall] - integralSum[ySmall][xLarge]
    mean = sum/((xLarge-xSmall)*(yLarge-ySmall))
    return mean

def calculateMeanArray(array):
    h, w = array.shape[0], array.shape[1]
    integralSumArray = integralSum(array)
    output = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            output[i][j] = calculateMean(integralSumArray,101,j,i,h,w)
    return output

def medianFilter(image, filterSize):
    temp = []
    index = filterSize // 2
    data_final = []
    finalData = np.zeros((len(image),len(image[0])))
    for i in range(len(image)):

        for j in range(len(image[0])):

            for z in range(filterSize):
                if i + z - index < 0 or i + z - index > len(image) - 1:
                    for c in range(filterSize):
                        temp.append(0)
                else:
                    if j + z - index < 0 or j + index > len(image[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filterSize):
                            temp.append(image[i + z - index][j + k - index])

            temp.sort()
            finalData[i][j] = temp[len(temp) // 2]
            temp = []
    return finalData

def thresholdWithoutOpenCV(image):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    imageArray = np.array(image)
    grayscale_image = np.dot(imageArray[..., :3], rgb_weights)
    mean = (calculateMeanArray(grayscale_image))
    difference = grayscale_image - mean
    k = 0.08
    threshold = mean * (1 + k * (difference / (1 - difference) - 1))
    grayscale_image = medianFilter(grayscale_image, 4)
    grayscale_image[grayscale_image > threshold] = 255
    grayscale_image[grayscale_image <= threshold] = 0
    return grayscale_image

def thresholdWithOpenCV(image):
    image = np.asarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    (T, threshInv) = cv2.threshold(blurred, 120, 255,cv2.THRESH_BINARY_INV)
    return threshInv

def connectedComponent(I):
    global label
    label = 1
    h, w = I.shape
    for Y in range(h):
        for X in range(w):
            if I[Y, X] != 0:
                I = assignLabel(X,Y,I)
    I = doUnion(h,w,I)
    return I

def assignLabel(X , Y, I):
    global label
    if Y == 0:
        if X == 0:
            I[Y, X] = label
        elif I[Y, X -1] != 0:
            I[Y, X] = I[Y, X -1]
    elif X == 0:
        if I[Y-1, X] != 0:
            I[Y, X] = I[Y-1, X]
    else:
        A = I[Y, X-1]
        B = I[Y-1, X]
        C = I[Y-1, X -1]
        valueArray = np.array([A,B,C])
        if(np.max(valueArray) !=0 ):
            I[Y,X] = np.min(valueArray[np.nonzero(valueArray)])
        else:
            label += 1
            I[Y, X] = label
    return I

def doUnion(h, w ,I):
    for Y in range(h):
        for X in range(w):
            if I[Y, X] != 0:
                for i in range(-1,2):
                    for j in range(-1, 2):
                        A = I[Y, X]
                        if ( -1 < (Y+i) < h) and (-1 < (X+j) < w):
                            B = I[Y+i,X+j]
                            if B != 0 and A != B:
                                if A < B:
                                    I[I == B] = A
                                else:
                                    I[I == A] = B
    return I

def connectedComponentAnalysis(image):

    #img = thresholdWithoutOpenCV(image)
    img = thresholdWithOpenCV(image)
    img = (img) // 255
    img = np.array(img).astype(int)
    img = connectedComponent(img)
    unique, counts = np.unique(img, return_counts=True)
    n = 0
    for i in unique:
        img[img == i] = n
        n += 1

    label_hue = np.uint8(179 * img / np.max(img))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return np.max(img)

if __name__ == '__main__':
    image = Image.open("demo4.png")
    result = connectedComponentAnalysis(image)
    print(result)