import random

def loadData(filename, castToFloats = False, shuffle = False,):
    with open(filename, "r") as textFile:
        ClassList = []
        FeatureList = []
        allValueList = []
        randomizeList = []

        for line in textFile:
            randomizeList.append(line)
        if (shuffle):
            random.shuffle(randomizeList)
        for line in randomizeList:
            classEmpty = True
            featureList = []
            valueList = []
            for feature in line.split(" "):
                if classEmpty:
                    ClassList.append(int(feature))
                    classEmpty = False
                elif feature == '\n':
                    classEmpty = True
                else:
                    featureSplit = feature.split(':')
                    if(castToFloats):
                        featureList.append(float(featureSplit[0]))
                        valueList.append(float(featureSplit[1]))
                    else:
                        featureList.append(featureSplit[0])
                        valueList.append(featureSplit[1])
            FeatureList.append(featureList)
            allValueList.append(valueList)
        return [ClassList, FeatureList, allValueList]


Data = loadData('LinearDataExample.txt')
ClassData = Data[0]
FeatureData = Data[1]
ValueData = Data[2]

def processData(ValueOnly = False, binaryFeatures = None):
    combinedFeatureValue = []
    questionLocator = ['29JqQTHzP0XcLZMmpZkc', '80r3q6g6UX6fTfzytMex', 'CMWt5vtu0FOahsOICqan',
                           'KYYz2ik0PGifMHOPCAKK', 'PYbZBDYyL1uR8kkUiIrc', 'XgRKqdCySNfbkm4bQqOJ',
                           'metuO2XDAtavCtZFnUw5', 'nQtoAvCJg4FgTh7Gk3qw', 'nRapwzuudOauSQlNN2L6',
                           't3tnn7xuguNLJaCAFjhs', 'wKCAR6c4yOm1x5s79dWm']
    combinedQuestionScores = [0] * len(questionLocator)
    combinedQuestionCount = [0] * len(questionLocator)
    combinedQuestionMean = [0] * len(questionLocator)
    for i, featureList in enumerate(FeatureData):
        newfeatureValueList = []
        valueList = ValueData[i]
        uniqueQuestions = ['29JqQTHzP0XcLZMmpZkc', '80r3q6g6UX6fTfzytMex', 'CMWt5vtu0FOahsOICqan',
                           'KYYz2ik0PGifMHOPCAKK', 'PYbZBDYyL1uR8kkUiIrc', 'XgRKqdCySNfbkm4bQqOJ',
                           'metuO2XDAtavCtZFnUw5', 'nQtoAvCJg4FgTh7Gk3qw', 'nRapwzuudOauSQlNN2L6',
                           't3tnn7xuguNLJaCAFjhs', 'wKCAR6c4yOm1x5s79dWm']

        for i, feature in enumerate(featureList):
            questionIndex = questionLocator.index(feature)
            combinedQuestionScores[questionIndex] += int(valueList[i])
            combinedQuestionCount[questionIndex] += 1
            newfeatureValueList.append((valueList[i], feature))
            if (uniqueQuestions.__contains__(feature)):
                uniqueQuestions.remove(feature)
        for i, score in enumerate(combinedQuestionScores):
            if(combinedQuestionCount[i] != 0):
                combinedQuestionMean[i] = score / combinedQuestionCount[i]
            else:
                combinedQuestionMean[i] = random.randint(0,5)
        for feature in uniqueQuestions:
            questionIndex = questionLocator.index(feature)
            generatedValue = combinedQuestionMean[questionIndex]
            if binaryFeatures != None:
                if generatedValue < binaryFeatures:
                    generatedValue = -1
                else:
                    generatedValue = 1
            newfeatureValueList.append((generatedValue, feature))
        sortedList = sorted(newfeatureValueList, key=lambda x: x[1])


        if(ValueOnly == True):
            newValueonlyList = []
            for list in sortedList:
                newValueonlyList.append(int(list[0]))
            sortedList = newValueonlyList
        combinedFeatureValue.append(sortedList)


    return combinedFeatureValue, combinedQuestionMean

combinedFeatureValue, combinedQuestionMean = processData(True)