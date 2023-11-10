# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Upload and read the dataset. 
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy. 
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
```

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:   BASKARAN  V
RegisterNumber:  212222230020

import pandas as pd
df=pd.read_csv("/content/Employee[1].csv")
df

df.head()

df.info()

df.describe()

df.isnull().sum()

df['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['salary']=le.fit_transform(df['salary'])
df

x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
y=df['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)



from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy


dt.predict([[0.92,0.85,5,259,5,0,0,1]])
```



## Output:
![image](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118703522/4807bfbe-3813-4f66-83e1-4add2ada5ff4)

![image](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118703522/88fb7207-2d4c-4ab6-a636-0814e33e7b78)

![image](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118703522/4ed0d74e-09a8-4b0f-abc4-bc0bae496329)


![image](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118703522/05bcb532-2da8-4e2b-a83c-7a10e3b2019a)


![image](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118703522/9bb6feb4-6190-4391-85b3-863078fa569d)

![image](https://github.com/BaskaranV15/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118703522/10207dfb-42a1-4091-9fb1-030a0a7317ab)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
