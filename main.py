
import pandas as pd

import predict
import preprocess
import verify

full_data = pd.read_csv("train.csv")
full_label = full_data["Survived"]
full_data = full_data.drop("Survived", axis=1)

# Preprocess
full_data = preprocess.fillNa(full_data)
full_data = preprocess.preprocessData(full_data, ["Sex", "Embarked", "Name", "Cabin", "Ticket"])
print "original:"
verify.verifyAccuracy(full_data, full_label)

# Remove some column to see if it works better
data_columns_removed = full_data.drop(["Cabin", "Fare", "Ticket", "Name", "PassengerId", "Parch", "SibSp", "Embarked"], axis = 1)
print "some_columns_removed"
verify.verifyAccuracy(data_columns_removed, full_label)