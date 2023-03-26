from sklearn.preprocessing import LabelEncoder


def dataCleaning(data):
    data_cleaned = data.dropna()
    return data_cleaned

def dataEncoding(data):
    objectList = data.select_dtypes(include = "object").columns.to_list()
    le = LabelEncoder()
    data_encoded = data.copy()
    for object in objectList:
        data_encoded[object] = le.fit_transform(data_encoded[object])

    return data_encoded