import cv2
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score,multilabel_confusion_matrix, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


dirName = os.path.dirname(__file__)
dataDir = dirName + "/Caltech20/"

numberOfClustersList = [50, 100,500]
numberOfClusters = numberOfClustersList[0]
descriptorNameList = ["SIFT", "SURF", "FAST", "ORB", "BRISK"]
descriptorName = descriptorNameList[0]
classifierList = ["SVM"]
classifierType = classifierList[0]
clusteringList = ["KMeans", "Spectral", "GMM", "MiniBatchKMeans"]
clustering = clusteringList[0]
kernelTypeList = ["Linear","Chi-Squared"]
kernelType = kernelTypeList[0]
CList = [ 10, 100,200,500]
C = CList[0]
trainingClassList = []
allF1Mean = np.zeros((len(numberOfClustersList)*len(kernelTypeList),len(CList)))
dataFrames = []


def getPaths(dataType="training"):
    ImagePaths = []
    subFolderPath = dataDir + dataType + '/'
    subfolderNames = os.listdir(subFolderPath)
    imagesWithClasses = []
    if (dataType == "training"):
        trainingClassList.clear()
    for subfolderName in subfolderNames:
        if (dataType == "training"):
            trainingClassList.append(subfolderName)
        imageNames = os.listdir(subFolderPath + subfolderName)
        ImagePathsEachClass = []
        for imageName in imageNames:
            ImagePathsEachClass += [(subFolderPath + subfolderName + '/' + imageName)]
            imagesWithClasses.append(subfolderName)
        ImagePaths += ImagePathsEachClass
    return ImagePaths, imagesWithClasses


trainingPaths, imageWithClassesTraining = getPaths()
testingPaths, imageWithClassesTesting = getPaths("testing")

def decideDescriptor(descriptorName):
    if descriptorName == "SIFT":
        return cv2.SIFT_create()
    elif descriptorName == "SURF":
        return cv2.SURF_create()
    elif descriptorName == "FAST":
        return cv2.FastFeatureDetector_create()
    elif descriptorName == "ORB":
        return cv2.ORB_create()
    elif descriptorName == "BRISK":
        return cv2.BRISK_create()

def detectAndCompute(descriptor, path):
    image = cv2.imread(path)
    return descriptor.detectAndCompute(image, None)[1]

def calculateDescriptions(descriptor, paths, imageWithClasses):
    descriptionList = []
    indexList = []
    for path in paths:
        description = detectAndCompute(descriptor, path)
        if description is not None:
            descriptionList.append(description)
        else:
            indexList.append(paths.index(path))
    for index in indexList:
        paths.pop(index)
        imageWithClasses.pop(index)
    return descriptionList, paths, imageWithClasses

def decideCluster(clustering):
    if clustering == "MiniBatchKMeans":
        return MiniBatchKMeans(n_clusters=numberOfClusters, max_iter= 50)
    elif clustering == "Spectral":
        return SpectralClustering(n_clusters=numberOfClusters)
    elif clustering == "KMeans":
        return KMeans(n_clusters=numberOfClusters)
    elif clustering == "GMM":
        return GaussianMixture(n_components=numberOfClusters)

def featureQuantization(descriptionList, clusteringResult, numberOfClusters):
    featureList = []
    for description in descriptionList:
        predictedResult = clusteringResult.predict(description)
        histogram, binEdges = np.histogram(predictedResult, bins=numberOfClusters)
        featureList.append(histogram)
    featureList = np.array(featureList)
    featureListNormalized = featureList / np.sum(np.abs(featureList), axis=1).reshape((featureList.shape[0], 1))
    return featureListNormalized

def BagOfWords(trainingPaths, testingPaths):
    descriptor = decideDescriptor(descriptorName)
    trainingDescriptions = calculateDescriptions(descriptor, trainingPaths)
    testingDescriptions = calculateDescriptions(descriptor, testingPaths)
    clusterObject = decideCluster(clustering)
    trainingClusteringResult = clusterObject.fit(np.concatenate(trainingDescriptions))
    featuresTrain = featureQuantization(trainingDescriptions, trainingClusteringResult, numberOfClusters)
    featuresTest = featureQuantization(testingDescriptions, trainingClusteringResult, numberOfClusters)
    return featuresTrain, featuresTest

def decideClassifier(classifierType):
    if classifierType == "SVM":
        return svm.SVC()

def decideKernel(kernelType):
    if kernelType == "Linear":
        return "linear"
    elif kernelType == "Chi-Squared":
        return chi2_kernel

class Classifier():

    def __init__(self, featuresTrain, classesTrain, featuresTest, classesTest):
        self.featuresTrain = featuresTrain
        self.classesTrain = classesTrain
        self.featuresTest = featuresTest
        self.classesTest = classesTest
        self.title = "Kernel: " + kernelType + ", Cluster Number: " + str(numberOfClusters) + ", C Parameter: " + str(C)
        self.path = dirName + "/results/" + "_".join(
            [str(numberOfClusters), descriptorName, classifierType, clustering, str(C), kernelType])
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def fit(self):
        kernel = decideKernel(kernelType)
        self.classifier = decideClassifier(classifierType)
        parameters = {'kernel': [kernel], 'C': [C]}
        scoring = {"Mean F1": make_scorer(f1_score, average="macro")}
        self.grid = GridSearchCV(self.classifier, parameters, scoring=scoring , cv = 5, refit="Mean F1")
        self.grid.fit(self.featuresTrain, self.classesTrain)


    def evaluate(self):
        self.predictedClasses = self.grid.predict(self.featuresTest)
        _, _, F1, _ = precision_recall_fscore_support(self.classesTest, self.predictedClasses, labels=np.unique(self.predictedClasses))
        F1Mean = np.mean(F1)
        self.conf_matrix = multilabel_confusion_matrix(self.classesTest, self.predictedClasses, labels=trainingClassList)
        with open(self.path + '/results.txt', 'w') as f:
            f.write("Mean F1 Score:" + str(F1Mean) + "\n")
            f.write("Per Class F1 Scores: \n")
            for i in range(len(np.unique(self.predictedClasses))):
                f.write(np.unique(self.predictedClasses)[i] + ": " + str(round(F1[i],2)) + "\n")
            f.write("Multi Label Confusion Matrix: \n")
            for i in range(len(trainingClassList)):
                f.write(trainingClassList[i] + ": \n[")
                for row in self.conf_matrix[i]:
                    f.write(str(row))
                f.write("]\n")
        titleList = np.array([self.title]*(len(trainingClassList)+1)).reshape(1,(len(trainingClassList)+1))
        differences = np.setdiff1d( np.unique(trainingClassList),np.unique(self.predictedClasses),)
        print(differences)
        differencesNP = np.array([differences, np.zeros(len(differences))])
        self.fScoresNP = np.array([np.unique(self.predictedClasses), np.around(F1,2)])
        self.fScoresNP = np.concatenate((self.fScoresNP, differencesNP), axis=1)
        self.fScoresNP = np.concatenate((self.fScoresNP, np.array((["Mean F1 Score"], [F1Mean]))), axis=1)
        self.fScoresNP = np.concatenate((titleList,self.fScoresNP),axis=0)
        df = pd.DataFrame(self.fScoresNP, index=["Title","Class", "Value"])
        dataFrames.append(df)
        df.to_csv(self.path +'/results.csv', sep = ';', float_format='%.2f', decimal=".", header = False)
        return F1Mean

    def saveConfusionMatrix(self, confusionMatrixNumber):
        self.confusionMatrix = confusion_matrix(self.classesTest, self.predictedClasses , labels = trainingClassList, normalize = 'true')
        self.df_cm = pd.DataFrame(self.confusionMatrix, index=trainingClassList,
                             columns=trainingClassList)
        fig, ax = plt.subplots(figsize=(13, 13))
        sn.heatmap(classifierObject.df_cm, annot=True, yticklabels=1, ax=ax, xticklabels=1)
        ax.set_title(self.title,fontsize =20)
        plt.savefig(self.path + '/confusionMatrix.png')
        plt.savefig('C:/Users/win10/Desktop/confff/' + str(confusionMatrixNumber) + '.png')
        plt.close()


    def saveMisclassifiedImages(self):
        misClassifiedIndices = np.where(np.array(self.predictedClasses != imageWithClassesTesting) == True)[0]
        dim = (20,20)
        heightImage = (len(misClassifiedIndices) // 20) + 1
        widthImage = 20
        canvas = np.ones((21 * heightImage - 1, 21 * widthImage - 1, 3), dtype=np.uint8) * 255
        if not os.path.exists(self.path + "/misClassifiedImages"):
            os.makedirs(self.path + "/misClassifiedImages")
        for i in misClassifiedIndices:
            index = np.where(misClassifiedIndices == i)[0][0]
            row = index//20
            column = index % 20
            image = cv2.imread(testingPaths[i])
            resizedImage = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            canvas[21*row:21*row+20, 21*column:21*column+20] = resizedImage
            filename = os.path.basename(testingPaths[i])
            cv2.imwrite(self.path + "/misClassifiedImages/" + filename, resizedImage)
        cv2.imwrite(self.path + "/misClassifiedImages/allTogether.jpg" , canvas)








if __name__ == '__main__':

    descriptor = decideDescriptor(descriptorName)
    trainingDescriptions, trainingPaths, imageWithClassesTraining = calculateDescriptions(descriptor, trainingPaths,
                                                                                          imageWithClassesTraining)
    testingDescriptions, testingPaths, imageWithClassesTesting = calculateDescriptions(descriptor, testingPaths,
                                                                                       imageWithClassesTesting)


    # Save calculated values
    # if not os.path.exists(dirName +"/npArray/"):
    #   os.makedirs(dirName +"/npArray/")
    # np.save(dirName +"/npArray/trainingDescriptions.npy", trainingDescriptions)
    # np.save(dirName +"/npArray/testingDescriptions.npy", testingDescriptions)
    # np.save(dirName +"/npArray/trainingPaths.npy", trainingPaths)
    # np.save(dirName +"/npArray/testingPaths.npy", testingPaths)
    # np.save(dirName +"/npArray/imageWithClassesTraining.npy", imageWithClassesTraining)
    # np.save(dirName +"/npArray/imageWithClassesTesting.npy", imageWithClassesTesting)


    # Load precalculated values
    # trainingDescriptions = np.load(dirName +"/npArray/trainingDescriptions.npy", allow_pickle=True)
    # testingDescriptions = np.load(dirName +"/npArray/testingDescriptions.npy", allow_pickle=True)
    # trainingPaths = np.load(dirName +"/npArray/trainingPaths.npy")
    # testingPaths = np.load(dirName +"/npArray/testingPaths.npy")
    # imageWithClassesTraining = np.load(dirName +"/npArray/imageWithClassesTraining.npy")
    # imageWithClassesTesting = np.load(dirName +"/npArray/imageWithClassesTesting.npy")

    confusionMatrixNumber = 1
    allF1MeanIndices = []
    for i in kernelTypeList:
        for j in numberOfClustersList:
            iterationName = "Kernel: " + str(i) + ", Cluster Number: " + str(j)
            allF1MeanIndices.append(iterationName)
            for k in CList:
                print(i,j,k)
                kernelType = i
                C = k
                numberOfClusters = j
                clusterObject = decideCluster(clustering)
                trainingClusteringResult = clusterObject.fit(np.concatenate(trainingDescriptions))
                print(i,j,k)

                featuresTrain = featureQuantization(trainingDescriptions, trainingClusteringResult, numberOfClusters)
                featuresTest = featureQuantization(testingDescriptions, trainingClusteringResult, numberOfClusters)
                classifierObject = Classifier(featuresTrain, imageWithClassesTraining,featuresTest, imageWithClassesTesting)
                print(i,j,k)

                classifierObject.fit()
                F1Mean = classifierObject.evaluate()
                allF1Mean[kernelTypeList.index(i)*len(numberOfClustersList)+numberOfClustersList.index(j),CList.index(k)] = F1Mean
                classifierObject.saveConfusionMatrix(confusionMatrixNumber)
                confusionMatrixNumber += 1
                print(i,j,k)
    dfMeanF1 = pd.DataFrame(allF1Mean, index=allF1MeanIndices,columns=["C: " + str(value) for value in CList])
    dfMeanF1.to_csv(dirName + '/F1MeanResults.csv', sep=';', float_format='%.2f', decimal=".")
    dfPerClass = pd.concat(dataFrames)
    dfPerClass.to_csv(dirName + '/F1PerClassResults.csv', sep=';', float_format='%.2f', decimal=".")


    # The best model
    # kernelType = kernelTypeList[1]
    # C = 500
    # numberOfClusters = 100
    # clusterObject = decideCluster(clustering)
    # trainingClusteringResult = clusterObject.fit(np.concatenate(trainingDescriptions))
    #
    # featuresTrain = featureQuantization(trainingDescriptions, trainingClusteringResult, numberOfClusters)
    # featuresTest = featureQuantization(testingDescriptions, trainingClusteringResult, numberOfClusters)
    # classifierObject = Classifier(featuresTrain, imageWithClassesTraining, featuresTest, imageWithClassesTesting)
    #
    # classifierObject.fit()
    # a = classifierObject.evaluate()
    # classifierObject.saveMisclassifiedImages()
    # print(a)



