# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required packages and read the data file.
2. Use LabelEncoder to convert categorical data into numerical data.
3. Split data into testing data and training data.
4. Apply Decision Tree Regressor.
5. Calculate mean squared error and R2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Keerthika N
RegisterNumber: 212221230049
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['Position']=le.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
* data.head()

![1](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427089/32ec3dda-81f0-4f6a-a216-d7a8a6823af0)

* data.info()

![2](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427089/93c02908-c290-4ae5-b3a3-e2ec6ccf8528)

* isnull() and sum()

![3](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427089/2ec8ccba-c0ad-4c2c-9654-9b91afa6c677)

* data.head() for salary

![4](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427089/bf50eb06-019a-4091-865d-a7a26564e27b)

* MSE value

![5](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427089/3ae07b24-c8ac-4f69-a302-fc36f0408a89)

* r2 value

![6](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427089/5179e826-4a03-465d-95f0-6f5c5e416631)

* data prediction

![7](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93427089/e5a4e8dd-95dd-4224-8030-6f805ce1c80a)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
