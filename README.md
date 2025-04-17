## NAME: SUJITHRA.K
## REGISTER NUMBER:212223040212

# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import pandas module and import the required data set.
   
2 .Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.


## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

```

## Output:
![image](https://github.com/user-attachments/assets/39791d22-9c3e-4673-bed3-42fbe717578c)
![image](https://github.com/user-attachments/assets/6580ad73-b0d5-4c3a-9140-cbeebe7ca931)
![image](https://github.com/user-attachments/assets/682ba515-a94e-44f1-ba59-f595b8b58ae1)
![image](https://github.com/user-attachments/assets/0329d0ca-d380-4981-8609-4bd89092fbb4)
![image](https://github.com/user-attachments/assets/2ea897bb-da9d-4d02-b41d-eb39e07a7047)
![image](https://github.com/user-attachments/assets/a52e491d-0db8-45aa-873c-487e815ae906)
![image](https://github.com/user-attachments/assets/bccaab4a-faec-4ebe-88cb-75b75f56b439)
![image](https://github.com/user-attachments/assets/a3e9ce2f-df36-4a20-84be-f0b7dc1277e9)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
