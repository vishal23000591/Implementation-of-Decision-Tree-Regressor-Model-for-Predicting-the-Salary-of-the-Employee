# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vishal S
RegisterNumber:  212223110063
*/


import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
## data.head():
![Screenshot 2024-04-02 092112](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/57012603-b542-4cee-9ecf-aaf749ee1638)
## data.info():
![Screenshot 2024-04-02 092201](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/0e5d079f-691c-4ebf-ac43-687269aae3ff)
## data.isnull().sum():
![Screenshot 2024-04-02 092231](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/d3d2b7a6-17e9-479b-b061-a518b6466b9d)
## data.head():
![Screenshot 2024-04-02 092305](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/7d37b16e-1b84-41e6-aa04-a19ab27a1c67)
## x.head():
![Screenshot 2024-04-02 092335](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/cf433ab7-28c3-4f09-b297-067fafb117dc)
## mse:
![Screenshot 2024-04-02 092401](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/5df6bedb-f201-40d1-a5cf-6b50db321a4e)
## r2:
![Screenshot 2024-04-02 092430](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/da02cb93-f8c5-4307-9ed1-fec0f79f313e)
## dt.predict([[5,6]]):
![Screenshot 2024-04-02 092450](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139719/4a4cddb9-574c-4959-9c28-339d8ef660c1)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
