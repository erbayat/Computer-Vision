import numpy as np
from PIL import Image, ImageFilter
from matplotlib import cm,pyplot as plt
import os
import cv2

numberOfPoints = 12
photoNameList = ['cmpeBuilding', 'northCampus']
photoName = photoNameList[1]
dirname = os.path.dirname(__file__)
relativePath = os.path.join(dirname, photoName + '\\')


def pickPointsManually(imgLeft,imgRight):
    fig = plt.figure()
    fig.set_size_inches(10, 7)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(imgLeft)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(imgRight)
    points = plt.ginput(2*numberOfPoints,timeout= 300, show_clicks=True)
    pointsLeft, pointsRight = points[::2], points[1::2]

    plt.close(fig)
    return np.array([pointsLeft, pointsRight])

def FormMatrix(p1 , p2):
    Mat = np.array([[p1[0]*p2[2],p1[1]*p2[2],p1[2]*p2[2],0,0,0,-p1[0]*p2[0],-p1[1]*p2[0],-p1[2]*p2[0]],
                    [0,0,0,p1[0]*p2[2],p1[1]*p2[2],p1[2]*p2[2],-p1[0]*p2[1],-p1[1]*p2[1],-p1[2]*p2[1]]])
    return Mat

def normalizeData(points):
    mean = points.sum(axis=0)/numberOfPoints
    squareDistance = np.sum(np.square(points) - np.square(mean))
    meanDistance = np.sqrt(squareDistance/(2*numberOfPoints))
    normalizerMatrix = np.array([[1/meanDistance,0,-mean[0]/meanDistance],[0,1/meanDistance,-mean[1]/meanDistance],[0,0,1]])
    return normalizerMatrix

def ArrangeMatrices(points1,points2):
    for i in range(numberOfPoints):
        temp = FormMatrix(points1[i],points2[i])
        if (i==0):
            ArrangedMat =temp
        else:
            ArrangedMat= np.vstack((ArrangedMat,temp))
    return ArrangedMat

def solveEquation(points1,points2):
    A = ArrangeMatrices(points1,points2)
    u,s,vt = np.linalg.svd(A)
    indx = np.argmin(s)
    H = vt[indx,:]
    H = H.reshape(3,3)

    return H

def quintetMerge(leftmostImage,leftImage,middleImage,rightImage,rightmostImage,H1,H2,H3,H4):
    tripleImage, tripleShift =  tripleMerge(leftImage,middleImage,rightImage,H2,H3)
    quintetImage, theLastShift = tripleMerge(leftmostImage,tripleImage,rightmostImage,H2@H1,H3@H4, tripleShift)
    return quintetImage

def tripleMerge(leftImage,middleImage,rightImage,H1,H2, tripleShift = np.array([0,0])):
    doubleImage, previousShift = mergeImage(leftImage,middleImage,H1,tripleShift)
    previousShift = previousShift + tripleShift
    tripleImage, newShift = mergeImage(rightImage,doubleImage,H2,previousShift)
    tripleShift = previousShift + newShift
    return tripleImage, tripleShift

def mergeImage(image1,image2, H1 , previousShift = np.array([0,0])):
    image2NP = np.array(image2)
    Y2,X2,_ = image2NP.shape
    maxCoor, minCoor, warpedImage = warpImage(image1,H1)
    Ywarped, Xwarped, _ = warpedImage.shape
    shiftX = min(minCoor[0] - previousShift[0], 0)
    shiftY = min(minCoor[1] - previousShift[1], 0)
    resultX = max(maxCoor[0] - previousShift[0], X2) - shiftX
    resultY = max(maxCoor[1] - previousShift[1], Y2) - shiftY
    resultImage = np.zeros((resultY, resultX, 3), dtype=np.uint8)
    extendedImage1 = np.zeros((resultY, resultX, 3), dtype=np.uint8)
    extendedImage2 = np.zeros((resultY, resultX, 3), dtype=np.uint8)
    extendedImage1[(minCoor[1]-shiftY-previousShift[1]):(minCoor[1]-shiftY-previousShift[1]+Ywarped),(minCoor[0]-previousShift[0]-shiftX):(minCoor[0]-previousShift[0]-shiftX+Xwarped)] = warpedImage
    extendedImage2[(-shiftY):(Y2-shiftY),(-shiftX):(X2-shiftX)] = image2NP
    resultImage = np.remainder(np.maximum(extendedImage1,((extendedImage2).astype(float)*257)-256),256).astype(int)
   # resultImage = np.maximum(extendedImage1,(extendedImage2)).astype(int)


    return resultImage, [shiftX,shiftY]


def interpolateWithCV(img, H, maxCoor, minCoor, interpolate_type):
    T = np.array([[1, 0, -minCoor[0]], [0, 1, -minCoor[1]], [0, 0, 1]])
    HTrans = T.dot(H)
    X,Y = int(np.ceil(maxCoor[0] - minCoor[0])), int(np.ceil(maxCoor[1] - minCoor[1]))
    sourceCoordinates = np.zeros((Y,X,3))
    sourceCoordinates[:,:,0] = np.tile(range(X), (Y, 1))
    sourceCoordinates[:,:,1] = np.tile(np.array(range(Y)).reshape(Y,1), (1, X))
    sourceCoordinates[:,:,2] = np.ones((Y,X))
    HInverse= np.linalg.inv(HTrans)
    sourceCoordinates = (HInverse @ sourceCoordinates[:,:,:,np.newaxis])
    sourceCoordinates = (np.delete((sourceCoordinates/sourceCoordinates[:,:,2,None]),2,axis=2)).reshape(Y,X,2)
    sourceCoordinates = np.array(sourceCoordinates,np.float32)
    interpolatedImage = cv2.remap(img, sourceCoordinates, None, interpolate_type)
    return interpolatedImage




def interPolate(imageNP, maxCoor, minCoor, newImageCoordinates):

    interpolatedImage = np.zeros((maxCoor[1]-minCoor[1] ,maxCoor[0]-minCoor[0],3))
    coordinates = np.array([newImageCoordinates[:, :, 0] - minCoor[0], newImageCoordinates[:, :, 1] - minCoor[1]])
    coordinatesTopLeft = coordinates.astype(int)
    coordinatesBotRight = np.array([coordinates[0]+1 , coordinates[1]+1]).astype(int)
    coordinatesBotLeft =  np.array([coordinates[0] , coordinates[1]+1]).astype(int)
    coordinatesTopRight = np.array([coordinates[0]+1, coordinates[1]]).astype(int)
    coordinatesTopLeftDiff = coordinates - coordinatesTopLeft

    interpolatedImage[coordinatesTopLeft[1],coordinatesTopLeft[0],:] = imageNP * ((1 - coordinatesTopLeftDiff[0]) * (1 - coordinatesTopLeftDiff[1]))[:,:,np.newaxis]
    interpolatedImage[coordinatesTopRight[1],coordinatesTopRight[0],:] += imageNP * (coordinatesTopLeftDiff[0] * (1 - coordinatesTopLeftDiff[1]))[:,:,np.newaxis]
    interpolatedImage[coordinatesBotRight[1],coordinatesBotRight[0],:] += imageNP * (coordinatesTopLeftDiff[0] * coordinatesTopLeftDiff[1])[:,:,np.newaxis]
    interpolatedImage[coordinatesBotLeft[1],coordinatesBotLeft[0],:] += imageNP * ((1 - coordinatesTopLeftDiff[0]) * coordinatesTopLeftDiff[1])[:,:,np.newaxis]

    return interpolatedImage

def warpImage(image, H):
    X,Y = image.size
    imageNP = np.array(image)
    cornerCoor = np.array([[0,0,1],[X-1,0,1],[0,Y-1,1],[X-1,Y-1,1]])
    newCorners = (H @ cornerCoor.T).T
    newCorners = newCorners/newCorners[:,-1,None]
    maxCoor = np.amax(newCorners, axis = 0).astype(int) + 2
    minCoor = np.amin(newCorners, axis = 0).astype(int) - 2


    warpedImage = np.zeros((maxCoor[1]-minCoor[1],maxCoor[0]-minCoor[0],3), dtype=np.uint8)
    newImageCoordinates = np.zeros((Y,X,3))
    newImageCoordinates[:,:,0] = np.tile(range(X), (Y, 1))
    newImageCoordinates[:,:,1] = np.tile(np.array(range(Y)).reshape(Y,1), (1, X))
    newImageCoordinates[:,:,2] = np.ones((Y,X))
    newImageCoordinates = (H @ newImageCoordinates[:,:,:,np.newaxis])
    newImageCoordinates = (np.delete((newImageCoordinates/newImageCoordinates[:,:,2,None]),2,axis=2)).reshape(Y,X,2)
    #warpedImage[newImageCoordinates[:,:,1].astype(int) - minCoor[1] , newImageCoordinates[:,:,0].astype(int) - minCoor[0], :] = imageNP

    #warpedImage = cv2.resize(warpedImage, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    #warpedImage = interPolate(imageNP,maxCoor,minCoor,newImageCoordinates).astype(int)

    warpedImage = interpolateWithCV(imageNP, H, maxCoor, minCoor, cv2.INTER_CUBIC).astype(np.uint8)

    return maxCoor, minCoor, warpedImage



def calculateHomography(points1, points2, isNormalized = True):
    T1 = normalizeData(np.array(points1))
    T2 = normalizeData(np.array(points2))
    points1 = np.hstack((np.array(points1), np.ones((numberOfPoints, 1))))
    points2 = np.hstack((np.array(points2), np.ones((numberOfPoints, 1))))
    H = np.zeros((3,3))
    if isNormalized == True:
        normalizedPoints1 = (T1 @ points1.T).T
        normalizedPoints2 = (T2 @ points2.T).T
        normalizedH1 = solveEquation(normalizedPoints1, normalizedPoints2)
        H = np.linalg.inv(T2) @ normalizedH1 @ T1
    else:
        H = solveEquation(points1, points2)
    return H

def pickAndSavePoints(image1,image2,image3,image4,image5):
    pointsList = np.zeros((8,numberOfPoints,2))
    pointsList[0:2] = pickPointsManually(image1,image2)
    pointsList[2:4] = pickPointsManually(image2,image3)
    pointsList[4:6] = pickPointsManually(image3,image4)
    pointsList[6:8] = pickPointsManually(image4,image5)
    with open('pointList' + photoName + '.npy', 'wb') as f:
        np.save(f, pointsList)
    return pointsList

def readPoints():
    with open('pointList' + photoName + '.npy', 'rb') as f:
        pointsList = np.load(f)
    return pointsList

def formWrongMatches(pointsList, K):
    tempList = np.copy(pointsList)
    tempList[3,0:K//2,1] = tempList[3,0:K//2,1]+np.random.uniform(80,120)*np.random.choice([-1,1])
    tempList[3,0:K//2,0] = tempList[3,0:K//2,0]+np.random.uniform(80,120)*np.random.choice([-1,1])
    tempList[4,0:K//2+1,1] = tempList[4,0:K//2+1,1]+np.random.uniform(80,120)*np.random.choice([-1,1])
    tempList[4,0:K//2+1,0] = tempList[4,0:K//2+1,0]+np.random.uniform(80,120)*np.random.choice([-1,1])
    return tempList

if __name__ == '__main__':


    image1 = Image.open(relativePath + "left-2.jpg")
    image2 = Image.open(relativePath + "left-1.jpg")
    image3 = Image.open(relativePath + "middle.jpg")
    image4 = Image.open(relativePath + "right-1.jpg")
    image5 = Image.open(relativePath + "right-2.jpg")
    #pointsList = pickAndSavePoints(image1,image2,image3,image4,image5)
    pointsList = readPoints()




    # take only five points
    fivePointsList = np.copy(pointsList[:,0:5,:])

    # wrong matches
    pointsListWM3NN = formWrongMatches(pointsList,3)

    pointsListWM3 = formWrongMatches(pointsList,3)

    pointsListWM5 = formWrongMatches(pointsList,5)

    # add noise
    gaussianNoise = np.random.normal(0, 20, (8,numberOfPoints,2))
    pointsListNoisy = np.copy(pointsList) + gaussianNoise


    # five points
    numberOfPoints = 5;
    H2FP = calculateHomography(fivePointsList[2],fivePointsList[3])
    H3FP = calculateHomography(fivePointsList[5],fivePointsList[4])
    numberOfPoints = 12;

    # wrong matches
    # not normalized
    H2WM3NN = calculateHomography(pointsListWM3NN[2],pointsListWM3NN[3],False)
    H3WM3NN = calculateHomography(pointsListWM3NN[5],pointsListWM3NN[4],False)
    #normalized
    H2WM3 = calculateHomography(pointsListWM3[2],pointsListWM3[3])
    H3WM3 = calculateHomography(pointsListWM3[5],pointsListWM3[4])
    H2WM5 = calculateHomography(pointsListWM5[2],pointsListWM5[3])
    H3WM5 = calculateHomography(pointsListWM5[5],pointsListWM5[4])

    # noisy Homograpies
    noisyH2 = calculateHomography(pointsListNoisy[2],pointsListNoisy[3])
    noisyH3 = calculateHomography(pointsListNoisy[5],pointsListNoisy[4])
    noisyH2NN = calculateHomography(pointsListNoisy[2],pointsListNoisy[3],False)
    noisyH3NN = calculateHomography(pointsListNoisy[5],pointsListNoisy[4],False)

    H1 = calculateHomography(pointsList[0],pointsList[1])
    H2 = calculateHomography(pointsList[2],pointsList[3])
    H3 = calculateHomography(pointsList[5],pointsList[4])
    H4 = calculateHomography(pointsList[7],pointsList[6])
    # finalImage = quintetMerge(image1,image2,image3,image4,image5,H1,H2,H3,H4)
    # figure = plt.figure()
    # plt.title("3.1.1 Five Points")
    # plt.imshow(tripleMerge(image2,image3,image4,H2FP,H3FP)[0])
    # figure.savefig(relativePath + photoName + "3.1.1 Five Points" + ".jpeg")
    # figure = plt.figure()
    # plt.title("3.1.2 Twelve Points")
    # plt.imshow(tripleMerge(image2,image3,image4,H2,H3)[0])
    # figure.savefig(relativePath + photoName + "3.1.2 Twelve Points" + ".jpeg")
    # figure = plt.figure()
    # plt.title("3.2.1 Three Wrong Matches Without Normalization")
    # plt.imshow(tripleMerge(image2,image3,image4,H2WM3NN,H3WM3NN)[0])
    # figure.savefig(relativePath + photoName + "3.2.1 Three Wrong Matches Without Normalization" + ".jpeg")
    # figure = plt.figure()
    # plt.title("3.2.2 Three Wrong Matches With Normalization")
    # plt.imshow(tripleMerge(image2,image3,image4,H2WM3,H3WM3)[0])
    # figure.savefig(relativePath + photoName + "3.2.2 Three Wrong Matches With Normalization" + ".jpeg")
    # figure = plt.figure()
    # plt.title("3.2.3 Five Wrong Matches With Normalization")
    # plt.imshow(tripleMerge(image2,image3,image4,H2WM5,H3WM5)[0])
    # figure.savefig(relativePath + photoName + "3.2.3 Five Wrong Matches With Normalization" + ".jpeg")
    # figure = plt.figure()
    # plt.title("3.3.1 Noisy Without Normalization")
    # plt.imshow(tripleMerge(image2,image3,image4,noisyH2NN,noisyH3NN)[0])
    # figure.savefig(relativePath + photoName + "3.3.1 Noisy Without Normalization" + ".jpeg")
    # figure = plt.figure()
    # plt.title("3.3.2 Noisy With Normalization")
    # plt.imshow(tripleMerge(image2,image3,image4,noisyH2,noisyH3)[0])
    # figure.savefig(relativePath + photoName + "3.3.2 Noisy With Normalization" + ".jpeg")
    figure = plt.figure()
    plt.title("3.4 Panaroma")
    plt.imshow(quintetMerge(image1,image2,image3,image4,image5,H1,H2,H3,H4))
    figure.savefig(relativePath + photoName + "3.4 Panaroma" + ".jpeg")
    plt.show()








