
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('G:\Data Analysis\output.csv')
df=df.dropna()
X = df.iloc[:,26:27].values
y = df.iloc[:, 1].values
for i in range(len(y)):
    y[i]=int(y[i])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

import numpy as np
y_pred = regressor.predict(X_test)
for i in range(len(y_pred)):
    y_pred[i]=int(y_pred[i])
    
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

model1=sm.OLS(y_train,X_train)

result=model1.fit()

print(result.summary())


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Opennees toExp (Training set)')
plt.xlabel('Opennees toExp')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Opennees toExp (Test set)')
plt.xlabel('Opennees toExp')
plt.ylabel('Salary')
plt.show()