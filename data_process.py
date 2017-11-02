import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

descrip = {1:'ID',
         2:'Clump Thickness',
         3:'Uniformity of Cell Size',
         4:'Uniformity of Cell',
         5:'Marginal Adhesion',
         6:'Single Epithelial Cell Size',
         7:'Bare Nuclei',
         8:'Bland Chromatin',
         9:'Normal Nucleoli',
         10:'Mitoses',
         11:'Class'
         }

data = pd.read_csv('data/data.txt')

#columns = [list(range(1,12))]

#data.columns = columns
columns = []

for i in descrip.keys():
    name = descrip[i].split()
    name = [j[0] for j in name]
    name = ''.join(name)
    columns.append(name.upper())

data.columns = columns
data.drop('I', axis=1, inplace=True)

#print(len([i for i in data.BN i]))


#verify the index of missing values
tratable_columns = []
for i in range(len(data.dtypes)):
    if data.dtypes[i] == 'object':
        tratable_columns.append(columns[i+1])

# print(tratable_columns)



# #adicionar regex aqui
# #print(data[tratable_columns].values.tolist())
# print(data.BN.values.tolist().count('?'))
cleanData = data.copy()
for i in range(len(data.BN)):
    if( data.BN[i] == '?'):
        cleanData.drop(i, axis = 0, inplace=True)

# print(data.shape)
# print(cleanData.shape)
cleanData.BN = cleanData.BN.astype('int64')
cleanData = cleanData[['CT','M','BN','BC','C']]
cleanDataAll = cleanData
# cleanData = cleanData.values
cleanData = cleanData.values.tolist()
cleanDataAll = cleanDataAll.values.tolist()

def split_data(percent, cleanData):
    data_len = len(cleanData)
    cleanData = np.array(cleanData)
    

    train = cleanData[:int(percent*data_len), :]
    test = cleanData[int(percent*data_len):, :]

    train_labels = train[:, -1]

    temp = []
    for i in train_labels:
        temp.append([1,0]) if i == 2 else temp.append([0,1])

    train_labels = np.array(temp)

    test_labels = test[:, -1]

    temp = []
    for i in test_labels:
        temp.append([1,0]) if i == 2 else temp.append([0,1])

    test_labels = np.array(temp)
    

    bias = np.ones(len(train))
    train_features = train[:, :-1]
    train_features = np.c_[train_features, bias]

    bias = np.ones(len(test))
    test_features = test[:, :-1]
    test_features = np.c_[test_features, bias]

    return train_features, train_labels, test_features, test_labels

train_features, train_labels, test_features, test_labels = split_data(0.9, cleanData)

# temp = []

# for i in range(len(cleanData)):
#     if cleanData[i][-1] == 2:
#         cleanData[i][-1] = [1,0]
#     else:
#         cleanData[i][-1] = [0,1]

