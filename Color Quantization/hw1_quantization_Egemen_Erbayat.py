import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.color import lab2rgb


def calculateDistance(colorOfPoints):
    distances = np.sqrt(((imageArray - colorOfPoints[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

def pickPointsRandomly():
    points = []
    for i in range(K):
        points.append((np.random.uniform(0,width),np.random.uniform(0,height)))
    return points

def pickPointsManually(img):
    plt.imshow(img)
    points = plt.ginput(K, show_clicks=True)
    return points

def getColor(img,points):
    colorOfPoints = np.zeros((K,3)) - 1
    for i in range(K):
        Coordinates = (int(points[i][0]), int(points[i][1]))
        color = np.array(img[Coordinates[1]][Coordinates[0]])
        for prevColor in colorOfPoints:
            if (np.array_equal(color, prevColor)):
                color = [np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)]
        colorOfPoints[i] = color
    return colorOfPoints

def iterate(colorOfPoints):
    return calculateDistance(colorOfPoints)

def findMean():
    return np.array([imageArray[nearestPoints==k].mean(axis=0) for k in range(K)]).astype(int)

def kMeans(colorOfPoints):
    global nearestPoints, meanColors
    nearestPoints = iterate(colorOfPoints)
    prevNearestPoints = [-1]*len(imageArray)
    iterationCount = 10
    for i in range(iterationCount):
        if(np.array_equal(prevNearestPoints,nearestPoints)):
            break
        meanColors = findMean()
        prevNearestPoints = nearestPoints
        nearestPoints = iterate(meanColors)


def createOutputImage(mode):
    outputImage = meanColors[nearestPoints]
    outputImage = np.reshape(outputImage,(-1,width,3))
    if mode == 'LAB':
        outputImage = lab2rgb(outputImage/500)*255*255
    image = Image.fromarray((outputImage).astype(np.uint8))
    image.save("1111_" + str(K) +"_" + str(mode) + "_manual.jpg")
    return image


def quantize(img, colorNumber):
    mode = 'RGB'
    global width, height, K, imageArray
    width, height = img.size
    K = colorNumber
    imageArray2D = np.array(img)
    imageArray = imageArray2D.reshape(-1, 3)
    if mode == 'LAB':
        imageArray = rgb2lab(imageArray/255)*500
    imageArray2D = np.reshape(imageArray,(-1,width,3))
    #points = pickPointsRandomly()
    points = pickPointsManually(img)
    colorOfPoints = getColor(imageArray2D,points)
    kMeans(colorOfPoints)
    return createOutputImage(mode)


def rgb2lab( imageArray ) :

    for i in range(len(imageArray)):
       num = 0
       RGB = [0, 0, 0]
       inputColor = imageArray[i]
       for value in inputColor:

           value = float(value) / 255

           if value > 0.04045 :
               value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
           else :
               value = value / 12.92

           RGB[num] = value * 100
           num = num + 1

       XYZ = [0, 0, 0,]

       X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
       Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
       Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
       XYZ[ 0 ] = round( X, 4 )
       XYZ[ 1 ] = round( Y, 4 )
       XYZ[ 2 ] = round( Z, 4 )

       XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
       XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
       XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

       num = 0
       for value in XYZ :

           if value > 0.008856 :
               value = value ** ( 0.3333333333333333 )
           else :
               value = ( 7.787 * value ) + ( 16 / 116 )

           XYZ[num] = value
           num = num + 1

       Lab = [0, 0, 0]

       L = ( 116 * XYZ[ 1 ] ) - 16
       a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
       b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

       Lab [ 0 ] = round( L, 4 )
       Lab [ 1 ] = round( a, 4 )
       Lab [ 2 ] = round( b, 4 )
       imageArray[i]=Lab
    return imageArray

if __name__ == '__main__':
    image = Image.open("img6.jpg")
    quantizedImage = quantize(image,2)
    plt.figure()
    plt.imshow(quantizedImage)
    plt.show()




