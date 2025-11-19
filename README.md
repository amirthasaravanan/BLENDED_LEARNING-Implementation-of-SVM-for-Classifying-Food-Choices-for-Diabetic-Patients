# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data Import and prepare the dataset to initiate the analysis workflow.

2.Explore Data Examine the data to understand key patterns, distributions, and feature relationships.

3.Select Features Choose the most impactful features to improve model accuracy and reduce complexity.

4.Split Data Partition the dataset into training and testing sets for validation purposes.

5.Scale Features Normalize feature values to maintain consistent scales, ensuring stability during training.

6.Train Model with Hyperparameter Tuning Fit the model to the training data while adjusting hyperparameters to enhance performance.

7.Evaluate Model Assess the model’s accuracy and effectiveness on the testing set using performance metrics.

## Program:
```
Program to implement SVM for food classification for diabetic patients.
Developed by: Amirtha Varshini M
RegisterNumber: 212224230017
```
```python
#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

#step-1:Load the dataset from the url
data=pd.read_csv("/content/food_items_binary (1).csv")

#step-2:Data Exploration
#Display the first few rows and the colum names for verification
print(data.head())
print(data.columns)

#step-3:Selection Features and Target
#Define relevant features and target column
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Monounsaturated Fat']
target = 'class'

x = data[features]
y = data[target]

#step-4: Splitting Data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


#step-5: Feature Scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#step-6:Model training with Hyperparameter Tuning using GridSearchCV
#Define the SVM model
svm=SVC()

#set up heperparameter grid for tuning
param_grid = {
    'C': [0.1,1,10,100],
    'kernel': ['linear','rbf'],
    'gamma': ['scale','auto']
}

#Initialize GridSearchCV
grid_search=GridSearchCV(svm,param_grid,cv=5)
grid_search.fit(x_train,y_train)

#Extract the best model
best_model=grid_search.best_estimator_
print("Name: AMIRTHA VARSHINI M")
print("Register Number: 212224230017")
print("Best Parameter:",grid_search.best_params_)

y_pred=best_model.predict(x_test)

accurancy=accuracy_score(y_test,y_pred)
print("Name: AMIRTHA VARSHINI M")
print("Register Number: 212224230017")
print("Accuracy:",accurancy)
print("Classification Report:\n",classification_report(y_test,y_pred))

#confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![WhatsApp Image 2025-10-29 at 18 57 30_153ae584](https://github.com/user-attachments/assets/d27552dd-6f21-4c79-8018-5d611bacf8c9)
![op ml 2](https://github.com/user-attachments/assets/ef39838c-ddbe-4142-88df-4af290118c0e)



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
