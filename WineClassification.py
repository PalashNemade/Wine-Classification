                                # Name - Palash Nemade
                                # ID - 1001329664
#Used the reference given by professor P.ipynb and scikit-learn.org

import pandas as pd                                         #To read data from csv file

from sklearn.model_selection import train_test_split        # Split data into training and test data

#from sklearn.tree import DecisionTreeClassifier            # Decision tree used initially as classification model.

from sklearn.ensemble import RandomForestClassifier         # Implemented model for classification

from sklearn.model_selection import cross_val_score         # To cross validate the accuracy

#Data file
wineDataFile = 'wine.csv'

#Read wine data from csv file and return Pandas DataFrame.
wineData = pd.read_csv(wineDataFile)

#Reserve percentage for training data and test data.
trainingDataPercent = 0.75
testDataPercent = 0.25

# Training and test accuracy array.
trainingAccuracy = []
testAccuracy = []

#Head data points of the whole dataframe.
#print (wineData.head())
#print(wineData.shape)

# "quality" class is to be predicted
classCol = 'quality'
# Attributes which cannot be used as classifier class, so use them as features.
featureCols = ['fixed acidity','volatile acidity','citric acid',\
                'residual sugar','chlorides','free sulfur dioxide', \
                'total sulfur dioxide','density','pH','sulphates','alcohol']

wineFeatures = wineData[featureCols]
wineClass = wineData[classCol]

#print(wineFeatures)
#print(list(wineClass))

trainFeature, testFeature, trainClass, testClass = \
    train_test_split(wineFeatures, wineClass, stratify=wineClass,
                     train_size=trainingDataPercent, test_size=testDataPercent)


#Create instance of the Random Forest Classifier.
#n_estimators which allows the number of trees in the forest is set to 125,
#this parameter helps in improving the accuracy of the model as its default value is 10.
#The max_features parameter allows to select maximum features out of given features when looking for best split.
randomForest = RandomForestClassifier(n_estimators=125,max_features=1)

#Here we build a forest tree using training set trainFeature and trainClass
randomForest.fit(trainFeature,trainClass)

#Get and store the training accuracy and test accuracy in respective arrays.
trainingAccuracy = randomForest.score(trainFeature, trainClass)
testAccuracy = randomForest.score(testFeature,testClass)

#Printing the training and test accuracy of the model.
print("Training set score for random forest: {:.3f}".format(trainingAccuracy))
print("Test set score for random forest: {:.3f}".format(testAccuracy))

#Here we start with the cross validation steps.
#At first place we get and store the prediction from the testFeature for the purpose of confusion matrix.
prediction = randomForest.predict(testFeature)

#Here we print the confusion matrix for the model which comes out to be a 6x6 matrix.
print("\n\nConfusion Matrix...")
print(pd.crosstab(testClass, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#This step includes calculating the cross validation score using the cross_val_score of the RandomForest classifier.
#As per instructions the cv parameter is set to 10.
CVScore = cross_val_score(randomForest, wineFeatures, wineClass, cv=10)

#Printing the cross validation score also its mean value over 10 folds.
print("\n\nCross-validation scores: {}".format(CVScore))
print("\nAverage cross-validation score: {:.3f}".format(CVScore.mean()))


#First implemention of the Decision Tree classifier on wine data.
#decisionTree = DecisionTreeClassifier()
#decisionTree.fit(trainFeature, trainClass)
#prediction = decisionTree.predict(testFeature)
#trainingAccuracy = decisionTree.score(trainFeature, trainClass)
#testAccuracy = decisionTree.score(testFeature,testClass)
#print("Training set score for decision tree: {:.3f}".format(trainingAccuracy)
#print("Test set score: {:.3f}".format(testAccuracy)