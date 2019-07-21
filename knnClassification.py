
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('G:\Data Analysis\output.csv')
#Job role and branch is taken into consideration
X = dataset.iloc[:, [4, 16]].values
#CTC is estimated
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2= LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
for i in range(len(y_test)):
    y_test[i]=int(y_test[i])
for i in range(len(y_train)):
    y_train[i]=int(y_train[i])


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
c=0
y_pred = classifier.predict(X_test)
for i in range(len(y_pred)):
    y_pred[i]=int(y_pred[i])
    y_pred[i]=(y_pred[i]<=y_test[i]+1  and y_pred[i]>=y_test[i]-1)
    if(y_pred[i]):
        c+=1   
acc=float(c/len(y_test))


