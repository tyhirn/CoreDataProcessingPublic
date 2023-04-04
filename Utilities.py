from ast import If
from mailbox import linesep
from sqlite3 import Timestamp
from statistics import mean, stdev
import firebase_admin
import numpy
import sklearn
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import MLUtilities
import random
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time

# datetime object containing current date and time
from PegosasAlgorithm import PegosasTrain

now = datetime.now()



# Function to download data from firestore and return the following within DownloadDataAnalytics dict:
# - "Time To Connect To DataBase" : the time in seconds it took to access the responses
# - "Number Of Responses" : This is the number of Responses accessed
# - "Question Key" : Provides a dictionary in the format {QuestionID: QuestionText}
# - "libSvm" : Provides either vanilla libsvm text file or our own libSvm file that contains user and date
# - "Data" : Contains a list of dict objects: {'User': '', 'DateTime': '','Class': '', 'Features' : [], 'Values':[]}
# Parameters:
# - downloadToComputer: if set to True, it will download the livsvm file to local computer Note path must be changed
# - VanillaLibSvm: if True, will format svm without user and date
# - startTime : TODO: Will add parameters to the data being collection specific start of timeFrame
# - endTime : TODO: '' end of timeFrame
# - user : TODO: will filter to one user.
def DownloadData(db,downloadToComputer = False, vanillaLibSvm = True, startTime = None, endTime = None, user = None):
    # This contains all of our return output
    DownloadDataAnalytics ={'Time To Connect To DataBase': 0, 'Number Of Responses': 0, 'Question Key': {}, 'libSvm String': '', 'Data': []}
    # DataBase Connection Start
    start_time = time.time()
    # Initialize the app with a service account, granting admin privileges
    # firebase_admin.initialize_app(cred)
    # db = firestore.client()
    # Collection in firestore.
    responses_ref = db.collection(u'SliderResponses')
    docs = responses_ref.stream()
    # DataBase Connection End
    DownloadDataAnalytics['Time To Connect To DataBase'] = str(time.time() - start_time) + ' seconds'
    libsvm = ''
    numberOfDocs = 0
    for doc in docs:
        numberOfDocs += 1
        # sorting Fields into respective variables:
        responsesString = u'{}'.format(doc.to_dict()['Responses'])
        moodRatingString = u'{}'.format(doc.to_dict()['MoodRating'])
        dateSubmittedString = u'{}'.format(doc.to_dict()['Date'])
        userIDString = u'{}'.format(doc.to_dict()['User'])
        # question ID's and their values are stored together so they must be split.
        #Data building:
        dataEntry = {'User': '', 'DateTime': '','Class': '', 'Features' : [], 'Values':[]}
        featureList = []
        valueList = []
        #String building: Our format for libsvm (Date User Class feature:value feature2:value2...\n) Vanilla libsvm is simply: class feature:value\n.
        if (vanillaLibSvm == False):
            libsvm += dateSubmittedString + ' '
            libsvm += userIDString + ' '
        libsvm += moodRatingString + ' '
        #var made for easy debugging.
        responseList = responsesString.split(', ')
        for responseElement in responseList:
            if responseElement == '':
                continue
            responseElementSplit = responseElement.split(':')
            # Features and values stored as strings
            questionIDFeature = responseElementSplit[0]
            questionResponseValue = responseElementSplit[1]
            libsvm += str(questionIDFeature) + ':' + str(questionResponseValue) + ' '
            # building feature and value lists for data entry
            featureList.append(str(questionIDFeature))
            valueList.append(str(questionResponseValue))
        libsvm += '\n'
        # crafting the dict entry:
        dataEntry['User'] = userIDString
        dataEntry['DateTime'] = dateSubmittedString
        dataEntry['Class'] = moodRatingString
        dataEntry['Features'] = featureList
        dataEntry['Values'] = valueList
        DownloadDataAnalytics['Data'].append(dataEntry)
    DownloadDataAnalytics['libSvm String'] = libsvm
    DownloadDataAnalytics['Number Of Responses'] = numberOfDocs
    # access question texts through firestore
    questions_ref = db.collection(u'TrainingQuestions')
    questionDocs = questions_ref.stream()
    for question in questionDocs:
        questionText = u'{}'.format(question.to_dict()['Question'])
        DownloadDataAnalytics['Question Key'][str(question.id)] = questionText

    questions2_ref = db.collection(u'TrainingQuestions06-07-22') #TODO: This is so bad.
    questionDocs = questions2_ref.stream()
    for question in questionDocs:
        questionText = u'{}'.format(question.to_dict()['Question'])
        DownloadDataAnalytics['Question Key'][str(question.id)] = questionText

    # Download straight to your computer, rememeber to change the path below
    if downloadToComputer:
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m%d%Y%H%M%S")
        with open('/Users/Tyler/Downloads/' + dt_string + '.txt', 'w') as f:
            f.write(libsvm)

    return DownloadDataAnalytics

# Function to process the data. The data is sorted
def ProcessData(dataInput = None, binaryMoodThreshold = None, dictBinaryFeatureValueThresholdsORValue = None, fillEmptyValues = None, randomizeEntries = False, classData = None, featureData = None, valueData = None, uniqueQuestionsList = None):
    userData = []
    if dataInput != None and type(dataInput) is str and dataInput.Contains('.txt'):
        # This is for using libsvm text files to reduce firestore usage
        Data = MLUtilities.loadData(dataInput)
        classData = Data[0]
        featureData = Data[1]
        valueData = Data[2]
    elif dataInput != None and type(dataInput[0]) is dict:
        # Assumption that if a dict is put in for dataInput, then the DownloadData() dict return is being used.
        classData = []
        featureData = []
        valueData = []
        userData = []
        Data = dataInput
        for dictElement in dataInput:
            classData.append(dictElement.get('Class'))
            featureData.append(list(x.strip() for x in dictElement.get('Features')))
            valueData.append(dictElement.get('Values'))
            userData.append(dictElement.get('User'))      
    # If uniqueQuestionsList is set, then it may reduce runtime to use a preprocessed list of questions.
    if uniqueQuestionsList == None:
        uniqueQuestionsList = set(featureData[0])
        for featureListElement in featureData:
            uniqueQuestionsList.extend(uniqueQuestionsList.difference(set(featureListElement)))
    elif type(uniqueQuestionsList) is dict:
        uniqueQuestionsList = list(uniqueQuestionsList.keys())
    # Process class Data if mood Threshold is set
    if binaryMoodThreshold != None:
        for i, classElement in enumerate(classData):
            if (int(classElement) >= binaryMoodThreshold):
                classData[i] = 1
            else:
                classData[i] = -1
    
    uniqueQuestionLength = len(uniqueQuestionsList)
    positiveCombinedQuestionScoresDict = dict.fromkeys(uniqueQuestionsList, 0)
    positiveCombinedQuestionCountDict = dict.fromkeys(uniqueQuestionsList, 0)
    positiveCombinedQuestionMeanDict = dict.fromkeys(uniqueQuestionsList, 0)
    negativeCombinedQuestionScoresDict = dict.fromkeys(uniqueQuestionsList, 0)
    negativeCombinedQuestionCountDict = dict.fromkeys(uniqueQuestionsList, 0)
    negativeCombinedQuestionMeanDict = dict.fromkeys(uniqueQuestionsList, 0)
    
    #we need to combined the features and values so we can sort them and maintain pairings.
    combinedData = []

    # this will go in in add an empty space for all missing feature/value pairs, and will sort the data by question ID.
    # This function will also keep track of the question counts and total values and fill in mean, if progressiveMean is set to true.
    for i in range(len(classData)):
        classVal = classData[i]
        feature = featureData[i]
        value = valueData[i]
        missingQuestions = list(set(uniqueQuestionsList).difference(set(featureData[i])))
        combinedDataElement = []
        for j in range(len(valueData[i])):
            if(fillEmptyValues is not None):
                if classVal >= 0:
                    positiveCombinedQuestionCountDict[feature[j]] += 1
                    positiveCombinedQuestionScoresDict[feature[j]] += int(value[j])
                else:
                    negativeCombinedQuestionCountDict[feature[j]] += 1
                    negativeCombinedQuestionScoresDict[feature[j]] += int(value[j])
            if dictBinaryFeatureValueThresholdsORValue == True:
                if int(value[j]) <= 1:
                    value[j] = -2
                elif int(value[j]) == 2:
                    value[j] = -1
                elif int(value[j]) == 4:
                    value[j] = 1
                elif int(value[j]) >= 5:
                    value[j] = 2
                else:
                    value[j] = 0
            combinedDataElement.append((feature[j], value[j]))
        if fillEmptyValues == 'progressiveMean':
            for question in uniqueQuestionsList:
            #Here we are filling in the mean of previous questions progressively, so that all the filled in means are not the same.
                if classVal >= 0:
                    if positiveCombinedQuestionCountDict[question] == 0:
                        positiveCombinedQuestionMeanDict[question] = random.randint(0,5) #THIS IS A HACK TO MAKE THE MEAN OF THE FIRST QUESTION BE 3.
                    else:
                        positiveCombinedQuestionMeanDict[question] = positiveCombinedQuestionScoresDict[question] / positiveCombinedQuestionCountDict[question]   
                else:
                    if negativeCombinedQuestionCountDict[question] == 0:
                        negativeCombinedQuestionMeanDict[question] = random.randint(0,5)
                    else:
                        negativeCombinedQuestionMeanDict[question] = negativeCombinedQuestionScoresDict[question] / negativeCombinedQuestionCountDict[question]
        #This loop will fill in the missing questions with None, unless fillEmptyValues is set to 'progressiveMean'.        
        for missingQuestionElement in missingQuestions:
            feature.append(missingQuestionElement)
            if fillEmptyValues == 'progressiveMean':
                if classVal >= 0:
                    combinedDataElement.append(positiveCombinedQuestionMeanDict[missingQuestionElement])
                else:
                    combinedDataElement.append(negativeCombinedQuestionMeanDict[missingQuestionElement])
            else:
                combinedDataElement.append((missingQuestionElement, 0))
        combinedData.append(combinedDataElement)
    
    combinedDataAndFeatures = []
    # if randomizeEntries is set to true, then the data will be randomized.
    if randomizeEntries != None:
        for i in range(len(combinedData)):
            # Sort combinedData by question ID
            combinedData[i].sort(key=lambda x: x[0])
            combinedDataAndFeatures.append((combinedData[i], classData[i], userData[i]))
        random.shuffle(combinedDataAndFeatures)        
        # now we separate back out classes, features, and values.
        lenFeatureValueList = len(combinedDataAndFeatures[0][0])
        for i in range(len(combinedDataAndFeatures)):
            featureData[i] = list((x[0].strip() for x in combinedDataAndFeatures[i][0]))
            valueData[i] = list((x[1] for x in combinedDataAndFeatures[i][0]))
            if len(featureData[i]) != lenFeatureValueList:
                print("Error: User data length does not match value data length.")
                return None
            if len(valueData[i]) != lenFeatureValueList:
                print("Error: User data length does not match value data length.")
                return None
            classData[i] = combinedDataAndFeatures[i][1]
            userData[i] = combinedDataAndFeatures[i][2]

    # #sort and return all data to its separate feature and value lists.
    # for i, dataElement in enumerate(combinedData):
    #     # sort each list by its question ID
    #     dataElement.sort(key=lambda tup: tup[0])
    #     newFeatureData.append([x[0] for x in dataElement])
    #     newValueData.append([x[1] for x in dataElement])
    

       
    return featureData, valueData, classData, userData

# TODO: I think I want a function that will turn the data into libSVM format. It can then be used in DownloadData.py. and elsewhere.

def ProcessDataforML(dataInputList, binaryMoodThreshold, uniqueQuestionsList):
    # Assumption that if a dict is put in for dataInput, then the DownloadData() dict return is being used.
    classData = []
    featureData = []
    valueData = []
    userData = []
    data = dataInputList
    for dictElement in dataInputList:
        classData.append(dictElement.get('Class'))
        featureData.append(list(x.strip() for x in dictElement.get('Features')))
        valueData.append(dictElement.get('Values'))
        userData.append(dictElement.get('User'))      

    uniqueQuestionsList = list(uniqueQuestionsList.keys())

    for i, classElement in enumerate(classData):
            if (int(classElement) >= binaryMoodThreshold):
                classData[i] = 1
            else:
                classData[i] = -1

    #we need to combined the features and values so we can sort them and maintain pairings.
    combinedData = []
    questionsUniqueToUser = {}
    uniqueUserList = set(userData)
    for user in uniqueUserList:
        questionsUniqueToUser[user] = []


    # this will go in in add an empty space for all missing feature/value pairs, and will sort the data by question ID.
    # This function will also keep track of the question counts and total values and fill in mean, if progressiveMean is set to true.
    for i in range(len(classData)):
        classVal = classData[i]
        feature = featureData[i]
        value = valueData[i]
        user = userData[i]
        missingQuestions = list(set(uniqueQuestionsList).difference(set(featureData[i])))
        combinedDataElement = []
        for j in range(len(valueData[i])):
            if int(value[j]) <= 1:
                value[j] = -2
            elif int(value[j]) == 2:
                value[j] = -1
            elif int(value[j]) == 4:
                value[j] = 1
            elif int(value[j]) >= 5:
                value[j] = 2
            else:
                value[j] = 0
            combinedDataElement.append((feature[j], value[j]))
            if questionsUniqueToUser[user].__contains__(feature[j]) == False:
                questionsUniqueToUser[user].append(feature[j])


        for missingQuestionElement in missingQuestions:
            feature.append(missingQuestionElement)
            combinedDataElement.append((missingQuestionElement, 0))
        combinedData.append(combinedDataElement)
    
    combinedDataAndFeatures = []
    # if randomizeEntries is set to true, then the data will be randomized.
    for i in range(len(combinedData)):
        # Sort combinedData by question ID
        combinedData[i].sort(key=lambda x: x[0])
        combinedDataAndFeatures.append((combinedData[i], classData[i], userData[i]))
    random.shuffle(combinedDataAndFeatures)        
    # now we separate back out classes, features, and values.
    lenFeatureValueList = len(combinedDataAndFeatures[0][0])
    for i in range(len(combinedDataAndFeatures)):
        featureData[i] = list((x[0].strip() for x in combinedDataAndFeatures[i][0]))
        valueData[i] = list((x[1] for x in combinedDataAndFeatures[i][0]))
        if len(featureData[i]) != lenFeatureValueList:
            print("Error: User data length does not match value data length.")
            return None
        if len(valueData[i]) != lenFeatureValueList:
            print("Error: User data length does not match value data length.")
            return None
        classData[i] = combinedDataAndFeatures[i][1]
        userData[i] = combinedDataAndFeatures[i][2]

    return featureData, valueData, classData, userData, questionsUniqueToUser 


# This function will download the data to a excel file.
def DownloadExcelFile(classData, featureData, valueData, uniqueQuestionDict, fileName = str(now), includeQuestionID = False):
    excelDict = {"Mood" : classData}

    if uniqueQuestionDict == None:
        return None #TODO: Maybe add support for this, better solution is just pull out the unique questions generator into its own function.
    
    for key in uniqueQuestionDict.keys():
        excelDict[key] = []
    
    #Ideally all this data would already be sorted making this redundant, but nevertheless.
    for i in range(len(classData)):
        for j, valueElement in enumerate(valueData[i]):
            #Check if the value is none, if so, add a empty string.
            if valueElement == None:
                valueElement = ''
            excelDict[featureData[i][j]].append(valueElement)

    for key in excelDict:
        if key == "Mood":
            continue
        newKey = uniqueQuestionDict[key]
        if includeQuestionID:
            newKey = key + ': ' + newKey
        excelDict[newKey] = excelDict[key]
        del excelDict[key]

    data = pd.DataFrame(excelDict)
    data.to_excel(r'/Users/Tyler/Downloads/' + fileName + '.xlsx', index=False)

def UploadWeightsForUserMixed(db, users, dataReturn):
    data = dataReturn.get('Data')
    if users == None:
        users = []
        for entry in data:
            if(not users.__contains__(entry['User'])):
                users.append(entry['User'])
    
    processedFeaturesList, processedValuesList, processedClassesList, processsedUsersList = ProcessData(dataInput=list(data), randomizeEntries=True, uniqueQuestionsList=dataReturn.get('Question Key'), binaryMoodThreshold=4, dictBinaryFeatureValueThresholdsORValue=True)
    
    for user in users:           
        userList = list(processsedUsersList)
        classList = list(processedClassesList)
        featuresList = list(processedFeaturesList)
        valuesList = list(processedValuesList)
        #Put the user data in the front of all the data.
        swapIndex = 0
        copiedUser = []
        copiedClasses = []
        copiedValues = []
        for i in range(len(classList)):
            if userList[i] == user:
                valuesList[i], valuesList[swapIndex] = valuesList[swapIndex], valuesList[i]
                classList[i], classList[swapIndex] = classList[swapIndex], classList[i]
                userList[i], userList[swapIndex] = userList[swapIndex], userList[i]
                swapIndex += 1
                copiedClasses.append(classList[i])
                copiedValues.append(valuesList[i])
                copiedUser.append(userList[i])
        classList.extend(copiedClasses)
        valuesList.extend(copiedValues)
        userList.extend(copiedUser)
        svc = SVC(kernel='linear')
        svc.fit(valuesList, classList)
        #rf = RandomForestClassifier(n_estimators=25)
        #rffit = rf.fit(valuesList, classList)
        #svcWeights = list(rffit.feature_importances_)  
        svcWeights = list(svc.coef_[0])

        weightDict = {}
        for index in range(len(svcWeights)):
            weightDict[featuresList[0][index]] = svcWeights[index]            

        stdWeightsDict = {}
        weightMean = mean(svcWeights)   
        stdDev = stdev(svcWeights)
        for i in range(len(svcWeights)):
            stdWeightsDict[featuresList[0][i]] = (svcWeights[i] - weightMean) / stdDev
        # get lowest weight in absolute value of stdWeightsDict.
        lowestWeight = min(stdWeightsDict.values(), key=abs)
        if lowestWeight == 0:
            # get second lowest absolute weight.
            lowestWeight = min(stdWeightsDict.values(), key=abs)[1]
            if lowestWeight == 0:
                # get third lowest absolute weight.
                lowestWeight = min(stdWeightsDict.values(), key=abs)[2]
                if lowestWeight == 0:
                    # get fourth lowest absolute weight.
                    lowestWeight = min(stdWeightsDict.values(), key=abs)[3]
                    if lowestWeight == 0:
                        lowestWeight = .5
        # divede all weights by lowest weight.
        for key in stdWeightsDict.keys():
            stdWeightsDict[key] = stdWeightsDict[key] / lowestWeight

        dbAdd = {u"userID": user, u"weights": weightDict, u"stdWeights": stdWeightsDict, u"created_at": datetime.now(), "created_by": "Tyler The Creator"}
        db.collection(u'Machine Learning').add(dbAdd)

def UploadWeightsForUser(db, users, dataReturn):
    data = dataReturn.get('Data')
    if users == None:
        users = []
        for entry in data:
            if(not users.__contains__(entry['User'])):
                users.append(entry['User'])
    
    processedFeaturesList, processedValuesList, processedClassesList, processsedUsersList, questionsUniqueToUser = ProcessDataforML(dataInputList=list(data), uniqueQuestionsList=dataReturn.get('Question Key'), binaryMoodThreshold=4)
    
    for user in users:           
        userList = list(processsedUsersList)
        classList = list(processedClassesList)
        featuresList = list(processedFeaturesList)
        valuesList = list(processedValuesList)


        userOnlyValuesList = []
        userOnlyClassesList = []
        userOnlyFeaturesList = []
        # Only use the data for the user
        for i, entry in enumerate(userList):
            if entry == user:
                userOnlyValuesList.append(valuesList[i])
                userOnlyClassesList.append(classList[i])
                userOnlyFeaturesList.append(featuresList[i])

        if len(userOnlyValuesList) < 9:
            continue


        
        svc = SVC(kernel='linear')
        svc.fit(userOnlyValuesList, userOnlyClassesList)
        #rf = RandomForestClassifier(n_estimators=25)
        #rffit = rf.fit(valuesList, classList)
        #svcWeights = list(rffit.feature_importances_)  
        svcWeights = list(svc.coef_[0])

        weightDict = {}
        for index in range(len(svcWeights)):
            weightDict[featuresList[0][index]] = svcWeights[index]            

        stdWeightsDict = {}
        weightMean = mean(svcWeights)   
        stdDev = stdev(svcWeights)
        for i in range(len(svcWeights)):
            stdWeightsDict[featuresList[0][i]] = (svcWeights[i] - weightMean) / stdDev
        # get lowest weight in absolute value of stdWeightsDict.
        lowestWeight = min(stdWeightsDict.values(), key=abs)
        if lowestWeight == 0:
            # get second lowest absolute weight.
            lowestWeight = min(stdWeightsDict.values(), key=abs)[1]
            if lowestWeight == 0:
                # get third lowest absolute weight.
                lowestWeight = min(stdWeightsDict.values(), key=abs)[2]
                if lowestWeight == 0:
                    # get fourth lowest absolute weight.
                    lowestWeight = min(stdWeightsDict.values(), key=abs)[3]
                    if lowestWeight == 0:
                        lowestWeight = .5
        # divede all weights by lowest weight.
        newstdWeightsDict = {}
        for key in stdWeightsDict.keys():
            stdWeightsDict[key] = stdWeightsDict[key] / lowestWeight
            if (questionsUniqueToUser[user].__contains__(key)):
                newstdWeightsDict[key] = stdWeightsDict[key]
        
        

        dbAdd = {u"userID": user, u"weights": weightDict, u"stdWeights": newstdWeightsDict, u"created_at": datetime.now(), "created_by": "Tyler The Creator"}
        db.collection(u'Machine Learning').add(dbAdd)

def TestData(dataReturn, test_size = .3, eliminate3= False, moodThreshold= 4, show_DT_plot = False, DT= True, svc = True):
    data = dataReturn.get('Data')

    if eliminate3 == True:
        originalSize = len(data)
        entriesRemoved = 0
        i = 0
        # Go through data and remove all entries with mood 3.
        while i < originalSize:
            if data[i]['Class'] == 3:
                data.pop(i)
                entriesRemoved += 1
            else:
                i += 1
        print("Removed " + str(entriesRemoved) + "/" + str(originalSize) + " entries from the data set.")
    
    processedFeaturesList, processedValuesList, processedClassesList, processsedUsersList = ProcessData(dataInput=list(data), randomizeEntries=True, uniqueQuestionsList=dataReturn.get('Question Key'), binaryMoodThreshold=moodThreshold, dictBinaryFeatureValueThresholdsORValue=True)
                
    
    users = []
    for entry in data:
        if(not users.__contains__(entry['User'])):
            users.append(entry['User'])

    for user in users:           
        userList = list(processsedUsersList)
        classList = list(processedClassesList)
        featuresList = list(processedFeaturesList)
        valuesList = list(processedValuesList)
        #Put the user data in the front of all the data.
        swapIndex = 0
        copiedUser = []
        copiedClasses = []
        copiedValues = []
        for i in range(len(classList)):
            if userList[i] == user:
                valuesList[i], valuesList[swapIndex] = valuesList[swapIndex], valuesList[i]
                classList[i], classList[swapIndex] = classList[swapIndex], classList[i]
                userList[i], userList[swapIndex] = userList[swapIndex], userList[i]
                swapIndex += 1
                copiedClasses.append(classList[i])
                copiedValues.append(valuesList[i])
                copiedUser.append(userList[i])
        classList.extend(copiedClasses)
        valuesList.extend(copiedValues)
        userList.extend(copiedUser)
        
        # # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(valuesList, classList, test_size=test_size, random_state=1, stratify= classList)

        print('User: ' + user)
        if DT:
            DT = DecisionTreeClassifier(max_depth= 3)
            DT.fit(X_train, y_train)
            ypredict = DT.predict(X_test)
            print('DT Accuracy: %.3f' % accuracy_score(y_test, ypredict))
            if show_DT_plot:
                plt.figure(figsize=(15, 10))
                questionKey = dataReturn.get('Question Key')
                # Use question key to get the correct question text from processedFeatureList.
                featureNames = []
                for feature in processedFeaturesList[0]:
                    featureNames.append(questionKey[feature][0:15])
                classNames = ['Bad Mood', 'Good Mood']
                sklearn.tree.plot_tree(DT, feature_names=featureNames, class_names=classNames, fontsize=15)
                plt.show()
        
        if svc:
            svc = SVC(kernel='linear')
            svc.fit(X_train, y_train)
            ypredict = svc.predict(X_test)
            print('SVC Accuracy: %.3f' % accuracy_score(y_test, ypredict))


