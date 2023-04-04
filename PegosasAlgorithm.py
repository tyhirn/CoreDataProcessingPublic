import MLUtilities
import numpy as np

def loadPegosasData():
    # Train Data
    TrainData = MLUtilities.loadData("a4a.txt", True)
    binaryTrainClassList = TrainData[0]
    binaryTrainFeatureList = TrainData[1]
    # Test Data
    TestData = MLUtilities.loadData("a4aTest.txt", True)
    binaryTestClassList = TestData[0]
    binaryTestFeatureList = TestData[1]

    processedTrainFeatures = MLUtilities.standardizeDataLists(binaryTrainFeatureList)
    processedTestFeatures = MLUtilities.standardizeDataLists(binaryTestFeatureList)


def PegosasTrain(classes, features, Lam, T = None):
    if T == None:
        T = len(classes)
    weights = [[0]*len(features[0])]
    t = 1

    for i in range(T):
        try:
            n = 1/(Lam*t)
            dotProduct = MLUtilities.vectorDotProduct(weights[i], features[i])
            const = 1 - (Lam * n)
            product = MLUtilities.multiplyVector(weights[i], const)
            if dotProduct < 1:
                const2 = (float(n)*float(classes[i]))
                multiplicationResult = MLUtilities.multiplyVector(features[i],const2)
                product = MLUtilities.addTwoVectors(multiplicationResult, product)
                weights.append(product)
            else:
                weights.append(product)
            t += 1
        except:
            print(i)
    return weights[t-1]

def TestData(trainingClasses, trainingFeatures, testingClasses, testingFeatures, Lambda):
    Correct = 0
    Incorrect = 0
    Weights = PegosasTrain(trainingClasses, trainingFeatures, Lambda)
    for i, X in enumerate(testingFeatures):
        Indicator = -1
        dotProduct = MLUtilities.vectorDotProduct(X, Weights)
        if dotProduct >= 0:
            Indicator = 1
        if Indicator == int(testingClasses[i]):
            Correct += 1
        else:
            Incorrect += 1
    # Checking the accuracy
    return Correct / (Correct + Incorrect)

# loadPegosasData()
# Weights = PegosasTrain(binaryTrainClassList, processedTrainFeatures, 2)
# print(TestData(binaryTrainClassList, processedTrainFeatures, binaryTestClassList, processedTestFeatures, 2))
