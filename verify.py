
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import predict

def predictScores(features_train, labels_train, features_test, labels_test):
    scores = []
    pred_naiveBayes = predict.predictByNaiveBayes(features_train, labels_train, features_test)
    scores.append(accuracy_score(labels_test, pred_naiveBayes))
    #pred_hand = predict.predictByHand(features_train, labels_train, features_test)
    #scores.append(accuracy_score(labels_test, pred_hand)) # accuracy:0.701473
    pred_dst = predict.predictByDecisionTree(features_train, labels_train, features_test)
    scores.append(accuracy_score(labels_test, pred_dst))
    pred_svm = predict.predictBySVM(features_train, labels_train, features_test)
    scores.append(accuracy_score(labels_test, pred_svm))
    pred_vote = predict.predictByVote([pred_naiveBayes, pred_dst, pred_svm])
    scores.append(accuracy_score(labels_test, pred_vote))
    return scores

def verifyAccuracy(full_data, full_labels):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    df = pd.DataFrame(columns=["NaiveBayes", "DecisionTree", "SVM", "vote"])
    for train_index, test_index in kf.split(full_data):
        features_train = full_data.take(train_index)
        labels_train = full_labels.take(train_index)
        features_test = full_data.take(test_index)
        labels_test = full_labels.take(test_index)

        scores = predictScores(features_train, labels_train, features_test, labels_test)
        df.loc[len(df)] = scores

    print df.mean()
