import random

def loadData(filename, shuffle = False):
    with open(filename, "r") as textFile:
        classList = []
        featureList = []
        allValueList = []
        randomizeList = []

        for line in textFile:
            randomizeList.append(line)
        if (shuffle):
            random.shuffle(randomizeList)
        for line in randomizeList:
            featureList = []
            valueList = []
            lineSplit = line.split(" ")
            
            for feature in lineSplit:
                if len(feature) == 1:
                    classList.append(int(feature))
                elif feature.Contains(':'):
                    featureSplit = feature.split(':')
                    featureList.append(featureSplit[0])
                    valueList.append(featureSplit[1])

            featureList.append(featureList)
            allValueList.append(valueList)
        return [classList, featureList, allValueList]

def Manhattan(p1, p2):
    dist = abs(float(p2) - float(p1))
    return dist

def vectorDotProduct(v1, v2):
    product = 0
    for i in range(len(v1)):
        product += float(v1[i]) * float(v2[i])
    return product

def addAndSubVector(constant, Vector, dir ='+'):
    newList = Vector
    for i, constant in enumerate(constant):
        if(dir == '-'):
            newList[i] = newList[i] - constant
        else:
            newList[i] = newList[i] + constant
    return newList

def multiplyVector(list,c):
    newList = list
    for i,x in enumerate(list):
        newList[i] = c * x
    return newList

def addTwoVectors(v1, v2):
    for i in range(len(v1)):
        v1[i] += v2[i]
    return v1

def splitData(classes, features, values, TrainingToTestRatio, numOfFeatures):
    # start by spliting the data for testing. Very ugly.
    trainingAmount = int(TrainingToTestRatio * len(classes) / (100 * numOfFeatures))
    testingAmount = int((100 - TrainingToTestRatio) * len(classes) / (100 * numOfFeatures))
    secondGroup = trainingAmount + testingAmount
    thirdGroup = secondGroup + trainingAmount + testingAmount
    thirdTestingInd = thirdGroup + trainingAmount
    secondTestingInd = secondGroup + trainingAmount

    trainingClasses = classes[:trainingAmount]
    trainingClasses.extend(classes[secondGroup:secondTestingInd])
    trainingClasses.extend(classes[thirdGroup:thirdTestingInd])

    trainingFeatures = features[:trainingAmount]
    trainingFeatures.extend(features[secondGroup:secondTestingInd])
    trainingFeatures.extend(features[thirdGroup:thirdTestingInd])

    trainingValues = values[:trainingAmount]
    trainingValues.extend(values[secondGroup:secondTestingInd])
    trainingValues.extend(values[thirdGroup: thirdTestingInd])

    testingClasses = classes[trainingAmount:secondGroup]
    testingClasses.extend(classes[secondTestingInd:thirdGroup])
    testingClasses.extend(classes[thirdTestingInd:])

    testingFeatures = features[trainingAmount:secondGroup]
    testingFeatures.extend(features[secondTestingInd:thirdGroup])
    testingFeatures.extend(features[thirdTestingInd:])

    testingValues = values[trainingAmount:secondGroup]
    testingValues.extend(values[secondTestingInd:thirdGroup])
    testingValues.extend(values[thirdTestingInd:])

    return [trainingClasses, trainingFeatures, trainingValues, testingClasses, testingFeatures, testingValues]

def standardizeDataLists(features, fill = 0, length = 0):
    if length == 0:
        length = len(features[0])
    result = []
    for feature in features:
        while (len(feature) < length):
            feature.append(fill)
        result.append(feature)
    return result

