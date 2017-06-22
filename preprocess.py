
from sklearn import preprocessing

def fillNa(full_data):
    full_data = full_data.fillna(0)
    return full_data

# Convert strings to numbers
def preprocessData(full_data, columns):
    for column in columns:
        le = preprocessing.LabelEncoder()
        le.fit(full_data[column])
        full_data[column] = le.transform(full_data[column])
    return full_data