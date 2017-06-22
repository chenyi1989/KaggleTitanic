
import pandas as pd
from sklearn import naive_bayes
from sklearn import svm

def predictByNaiveBayes(features, labels, test_features):
    clf = naive_bayes.GaussianNB()
    clf.fit(features, labels)
    return clf.predict(test_features)

def predictBySVM(features, labels, test_features):
    clf = svm.SVC()
    clf.fit(features, labels)
    return clf.predict(test_features)

from sklearn import tree
def predictByDecisionTree(features, labels, test_features):
    clf = tree.DecisionTreeClassifier(min_samples_split=4)
    clf.fit(features, labels)
    return clf.predict(test_features)

def predictByHand(features, labels, test_features):
    predictions = []
    for _, passenger in test_features.iterrows():
        if passenger['Sex'] == 0:
            if (passenger['Pclass'] == 3 and passenger['Age'] >= 30):
                predictions.append(0)
                continue
            predictions.append(1)
        elif (passenger['Sex' == 1]):
            if (passenger['Age'] < 10):
                predictions.append(1)
                continue
            predictions.append(0)

    return pd.Series(predictions)

def predictByVote(preds):
    result = []
    for index in range(len(preds[0])):
        vote = 0
        for pred in preds:
            if pred[index] >= 1:
                vote = vote + 1
        if (vote > len(preds) / 2):
            result.append(1)
        else:
            result.append(0)
    return result