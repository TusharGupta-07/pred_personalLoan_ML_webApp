#!/usr/bin/env python
# coding: utf-8

# **Marketing Campaign for Banking Products**

# **Objective**:
# 
# The classification goal is to predict the likelihood of a liability customer buying personal
# loans.

# **Some Basics Steps and tasks**:
# 1. Import the datasets and libraries, check datatype, statistical summary, shape, null
# values etc
# 2. Check if you need to clean the data for any of the variables
# 3. EDA: Study the data distribution in each attribute and target variable, share your findings.
# 
# ● Number of unique in each column?
# 
# ● Number of people with zero mortgage?
# 
# ● Number of people with zero credit card spending per month?
# 
# ● Value counts of all categorical columns.
# 
# ● Univariate and Bivariate analysis
# 
# 4. Apply necessary transformations for the feature variables
# 5. Normalise your data and split the data into training and test set in the ratio of 70:30
# respectively
# 6. Use the Logistic Regression model to predict the likelihood of a customer buying
# personal loans.
# 7. Print all the metrics related for evaluating the model performance
# 8. Build various other classification algorithms and compare their performance
# 9. Give a business understanding of your model

# Attribute Information: 
# 
# ● ID: Customer ID
# 
# ● Age: Customer's age in completed years
# 
# ● Experience: #years of professional experience
# 
# ● Income: Annual income of the customer ($000)
# 
# ● ZIP Code: Home Address ZIP code.
# 
# ● Family: Family size of the customer
# 
# ● CCAvg: Avg. spending on credit cards per month ($000)
# 
# ● Education: Education Level. 
# 
# 1: Undergrad; 
# 
# 2: Graduate; 
# 
# 3: Advanced/Professional
# 
# ● Mortgage: Value of house mortgage if any. ($000)
# 
# ● Personal Loan: Did this customer accept the personal loan offered in the last campaign?
# 
# ● Securities Account: Does the customer have a securities account with the bank?
# 
# ● CD Account: Does the customer have a certificate of deposit (CD) account with the bank?
# 
# ● Online: Does the customer use internet banking facilities?
# 
# ● Credit card: Does the customer use a credit card issued by the bank?

# 
# **IMPORTING LIBRARIES**
# 
# 1. importing libraries for checking datatype, statistical summary, shape, null values etc
# 
# 2. to clean up the noise of the data
# 
# 3. for Classification Algorithms
# 
# 4. for Confusion Matrix
# 
# 5. for Classification reports
# 
# and more.
# 
# 
# 

# In[50]:




# In[51]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
#sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
#MLPClassfier
from sklearn.neural_network import MLPClassifier
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split


# **UPLOAD FILES**

# In[52]:


#from google.colab import files
#uploaded = files.upload()


#  **LOAD DATA AS .xlsx**

# **FILE Source** : https://www.kaggle.com/itsmesunil/bank-loan-modelling

# In[53]:

import io


user_data = pd.read_excel('Bank_Personal_Loan_Modelling copy.xlsx',"Data")

#user_data = pd.read_csv('Bank_Personal_Loan_Modelling.csv',encoding= 'unicode_escape',error_bad_lines=False)

#user_data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]


# In[ ]:


#**INSPECTING DATA**


# In[ ]:


#user_data = pd.read_excel('Bank_Personal_Loan_Modelling copy.xlsx',"Data")
#user_data = pd.read_csv('/Users/tushar_gupta_mac/PycharmProjects/flaskProject/Bank_Personal_Loan_Modelling.csv')

#user_data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]


# **INSPECTING DATA**
# 

# **head() method is used to return top n (5 by default) rows of a data frame or series.**

# In[ ]:


#TOP 10 ROWS OF OUR DATA SET

user_data.head(10)


# **tail()** function returns last n rows from the object based on position.

# In[ ]:


#BOTTOM 10 ROWS OF OUR DATA SET
user_data.tail(10)


# **describe()** is used to view some basic statistical details like count, mean, std, min, max etc. of a data frame or a series of numeric values.

# In[ ]:


user_data.describe()


# **OBSERVATION :** describe()
# 
# 1. We observe experience is below 0, experience cannot be negative 
# 
# 2. Zip code is related to area,and also it contains random values, hence ignoring it for now
# 
# 3. People with higher income takes Personal Loan but people with lower income does not take loan.
# 
# 4. Data of personal loan is overlaped wrt to most variables.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


user_data.info()


# **SHAPE OF DATA**

# In[ ]:


user_data.shape


# **CHECKING DATA TYPES**

# In[ ]:


user_data.dtypes


# CHECKING NULL DATA IN THE .xlsx FILE. BUT, NO NULL DATA FOUND

# In[ ]:


user_data.isnull().sum()


# HOW MANY VALUES ARE BELOW ZERO

# In[ ]:


(user_data['Experience']<0).sum()


# we have found 52 values below 0, this is an error
# 
# ---
# 
# 

# We got negative values in Experience column in our data set so we need to clean this, we are replacing the negative values with the median.

# In[ ]:


mdn = int(user_data['Experience'].median()) #This will calculate median of column Experience
for i in range(len(user_data)):
    if user_data['Experience'].iloc[i] < 0:
        user_data['Experience'].iloc[i] = mdn


# In[ ]:


user_data['Experience'].describe()


# **-> corr(), Find the pairwise correlation of all columns in the dataframe. Any NAN values are automatically excluded and,for any non-numeric data type columns in the dataframe it is ignored.**

# In[ ]:


bank_corr=user_data.corr()
plt.subplots(figsize =(15, 10)) 
sns.heatmap(bank_corr,cmap="YlGnBu",annot=True)


# **OBSERVATION :**
# 
# We found that, 'Age' and 'Experience' are highly co-related, So, we can drop any of the variable.
# 
# 
# also, there is almost Negligible relation between Personal loan and ZIP Code.
# 
# and, 'CCAvg' and 'Income' also have high correlation

# In our data set we do not found any impact of 'ID' and 'Experience' column, so, this will considered as NOISY data so, we decided to drop it

# In[ ]:


user_data.drop(['ID','Experience'], axis=1,inplace=True) #axis=1 represents column
user_data.describe()


# **EDA - EXPLORATORY DATA ANALYSIS**

# **Unique values in each column**

# In[ ]:


#this will print all unique values in each column of our data set

uniqueValues = user_data.nunique()
print('Count of unique values in each column :')
print(uniqueValues)


# **NUMBER OF PEOPLE WITH ZERO MORTGAGE**

# In[ ]:


(user_data['Mortgage']==0).sum()


# **OBSERVATION :**
# 
# The number of people with zero mortgage are 3462, that means majority of people don’t have mortgage
# 

# **NUMBER OF PEOPLE WITH ZERO CREDIT CARD SPENDING PER MONTH**

# In[ ]:


(user_data['CCAvg']==0).sum()


# **OBSERVATION :**
# 
# Number of people with zero credit card spending per month is 106, that means thousands of people use their credit card for making transactions per month

# **VALUES COUNT OF ALL CATRGORICAL DATA**
# 
# A Categorical variable is a variable that can take on one of a limited, and usually fixed, number of possible values,
# 
# 

# In[ ]:


user_data.Family.value_counts()


# In[ ]:


user_data.Education.value_counts()


# In[ ]:


user_data['Securities Account'].value_counts()


# In[ ]:


user_data['CD Account'].value_counts()


# In[ ]:


user_data.Online.value_counts()


# In[ ]:


user_data.CreditCard.value_counts()


# GROUPING BASED ON Personal Loan, i.e., MEAN,MEDIAN 

# In[ ]:


user_data.groupby(['Personal Loan']).agg(['mean','median'])


# GROUPING BASED ON Personal Loan i.e., MAX, MIN

# In[ ]:


user_data.groupby(['Personal Loan']).agg(['min','max'])


# In[ ]:


user_data.describe()


# **UNIVARIATE :**
# 
# Univariate basically tells us how data in each feature is distributed and also tells us about central tendencies like mean, median, and mode.

# In[ ]:


sns.distplot( user_data['Age'], color = 'r')


# In[ ]:


sns.distplot( user_data[user_data['Personal Loan'] == 0]['CCAvg'], color = 'r')
sns.distplot( user_data[user_data['Personal Loan']==1]['CCAvg'], color = 'g')


# **OBSERVATION :**
# 
# Coustomers who took Personal Loan also have high CCAvg but who don't have Personal Loan don't have good CCAvg, conclusion is that, the person who have good CCAvg can take Personal Loan, and can can be a targated coustomer
# 
# **and also, this data is Right Skewed So, we need to Normalise the data;**
# 

# In[ ]:


sns.distplot( user_data[user_data['Personal Loan'] == 0]['Income'], color = 'r')
sns.distplot( user_data[user_data['Personal Loan'] == 1]['Income'], color = 'g')


# **OBSERVATION :**
# 
# Coustomers who took Personal Loan have high Income, and who don't took Personal Loan don't have high Income

# In[ ]:


sns.distplot( user_data[user_data['Personal Loan'] == 0]['Education'], color = 'r')
sns.distplot( user_data[user_data['Personal Loan'] == 1]['Education'], color = 'g')


# **OBSERVATION :**
# 
# At all Education levels people have took Personal Loan

# In[ ]:


sns.distplot( user_data[user_data['Personal Loan'] == 0]['Mortgage'], color = 'r')
sns.distplot( user_data[user_data['Personal Loan']==1]['Mortgage'], color = 'g')


# **OBSERVATION :**
# Customers who did not take personal loan Have a very low mortgage but who did take loan have very high mortgage
# 
# **and, the data is Right Skewed**

# In[ ]:


sns.pairplot(user_data,diag_kind='kde',hue='Personal Loan')


# **BIVARIATE**
# 
# Bivariate analysis is one of the simplest forms of quantitative analysis. It involves the analysis of two variables, for the purpose of determining the empirical relationship between them. Bivariate analysis can be helpful in testing simple hypotheses of association.

# In[ ]:


sns.boxplot(x='Personal Loan',y='Age',data=user_data)


# **OBSERVATION :**
# 
# Age does not impact much to the chance of a person who take personal loan 

# In[ ]:


sns.boxplot(x='Personal Loan',y='Income',data=user_data)


# **OBSERVATION :**
# 
# Person with higher Income take more Personal Loan, But person with lower income does not take Personal Loan

# In[ ]:


sns.boxplot(x='Personal Loan',y='CCAvg',data=user_data)


# **OBSERVATION :**
# 
# Person with higher CCAvg(credit card spending per month) take more Personal Loan, But person with lower CCAvg does not take Personal Loan

# In[ ]:


sns.boxplot(x="Education", y="Income", hue="Personal Loan", data=user_data)


# **OBSERVATION :**
# 
# Person with  education level 1 have higher incomes. But customers took personal loans have the same income distribution regardless of the education level.
# 
# 

# In[ ]:


sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=user_data)


# **OBSERVATION :**
# 
# Person who have Personal Loan have more mortgages
# 

# In[ ]:


Loan_zipcode = user_data[user_data['Personal Loan']==1]['ZIP Code'].value_counts().head(10)
Loan_zipcode


# **OBSERVATION :**
# 
# These are the top 10 locations from where the coustomers previously applied for Personal Loan

# In[ ]:


user_data_x=user_data.loc[:,user_data.columns !='Personal Loan']
user_data_y=user_data[['Personal Loan']]


# **TRANSFORMATIONS FOR FEATURE VARIABLES**

# As shown in UNIVARIATE graph above, that we got the right skewed data, so we had to transform it to the Normal distribution.
# We are using "yeo-johnson" from PowerTransformer 

# In[ ]:


power_t=PowerTransformer(method="yeo-johnson", standardize=False)
power_t.fit(user_data_x['Income'].values.reshape(-1,1))

user_data_x['Income']=power_t.transform(user_data_x.Income.values.reshape(-1,1))
sns.distplot(user_data_x['Income']);


# **OBSERVATION :**
# 
# Unlike before now the data is in Normal distribution

# In[ ]:


power_t=PowerTransformer(method="yeo-johnson", standardize=False)
power_t.fit(user_data_x['CCAvg'].values.reshape(-1,1))

user_data_x['CCAvg']=power_t.transform(user_data_x.CCAvg.values.reshape(-1,1))
sns.distplot(user_data_x['CCAvg']);


# We have decided to drop 'ZIP Code' column as it is not contribution in our program because it is just random values, and considered as noisy data, So, droping it, is a good idea.

# In[ ]:


user_data_x.drop(['ZIP Code'], axis=1,inplace=True) #axis=1 represents column
user_data_x.describe()


# In[ ]:


user_data_x['Mortgage_INT']=pd.cut(user_data_x['Mortgage'],
                                   bins=[0,100,200,300,400,500,600,700],
                                   labels=[0,1,2,3,4,5,6],include_lowest=True)
user_data_x.drop('Mortgage',axis=1,inplace=True)
sns.distplot(user_data_x['Mortgage_INT']);


# we have used binning in Mortgage to normalise Mortgage data.
# 
# **BINNING method** : 
# 
# Binning method is used to smoothing data or to handle noisy data. In this method, the data is first sorted and then the sorted values are distributed into a number of buckets or bins. As binning methods consult the neighborhood of values, they perform local smoothing.

# In[ ]:


user_data_x.head()


# **SPLITING DATA INTO TRAINING SET AND TEST SET IN 70:30 RATIO RESPECTIVELY**

# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(user_data_x,user_data_y,test_size=0.3,stratify=user_data_y,random_state=0)


# **SHAPE OF TEST SET AND TRAINING SET**

# In[ ]:


print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)


# In[ ]:


train_x.head()


# **USING DIFFERENT ALGORITHM TO CLASSIFY, WHICH ALGORITHM IS GIVING US THE BEST ACCURACY FOR OUR DATA SET**
# 
# 
# 
# 
# 

# reset_index() : Pandas reset_index() is a method to reset index of a Data Frame. reset_index() method sets a list of integer ranging from 0 to length of data as index.

# In[ ]:


train_x.reset_index(drop=True, inplace=True)
test_x.reset_index(drop=True, inplace=True)
train_y.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
for ind, column in enumerate (train_x.columns):
  scaler = StandardScaler()

  #fit to train data 
  scaler.fit(train_x[[column]])

  #transform train data
  np_array = scaler.transform(train_x[[column]])
  train_x.loc[:, column] = pd.Series(np_array.flatten())

  #transform test data
  np_array = scaler.transform(test_x[[column]])
  test_x.loc[:, column] = pd.Series(np_array.flatten())


# **Converting dataframes to numpy arrays**

# In[ ]:


np_train_x=train_x.values
np_train_y=train_y.values
np_test_x=test_x.values
np_test_y=test_y.values


# In[ ]:


np_test_y.shape


# **Learner Classifiers**

# In[ ]:


classifier_1 = LogisticRegression(random_state=0)
classifier_2 = DecisionTreeClassifier(random_state=0, max_depth=8)
classifier_3 = RandomForestClassifier(random_state=0, n_estimators=500,max_depth=8)


# **For Confusion Matrix**

# In[ ]:


def draw_cm(actual, predicted):
  cm = confusion_matrix(actual, predicted) 
  sns.heatmap(cm, annot=True, fmt='.2f', xticklabels = [0,1], yticklabels = [0,1] ) 
  plt.ylabel("Observed") 
  plt.xlabel("Predicted") 
  plt.show()


# **Logistic Regression** : 
# 
# Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary). Like all regression analyses, the logistic regression is a predictive analysis. Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. Sometimes logistic regressions are difficult to interpret; the Intellectus Statistics tool easily allows you to conduct the analysis, then in plain English interprets the output.

# In[ ]:


# fit classifier_1
classifier_1.fit(np_train_x, np_train_y.ravel())
pred_1_test_x = classifier_1.predict(np_test_x)
pred_1_train_x = classifier_1.predict(np_train_x)
accuracy_score_1_train_x=accuracy_score(np_train_y,pred_1_train_x)
accuracy_score_1_test_x=accuracy_score(np_test_y,pred_1_test_x)

print("Accuracy for train data : {:.4f} ".format(accuracy_score_1_train_x))
print("Accuracy for test data : {:.4f} ".format(accuracy_score_1_test_x))


# Confusion Matrix for classifier_1

# In[ ]:


print("Confusion Matrix : ")
print(draw_cm(np_test_y,pred_1_test_x.reshape(-1,1)))


# In[ ]:


print("Mean Absolute Error : ", metrics.mean_absolute_error(np_test_y,pred_1_test_x))
print("R2 Score : ",metrics.r2_score(np_test_y,pred_1_test_x))


# Classification Report :

# In[ ]:



print(classification_report(np_test_y,pred_1_test_x))


# **OBSERVATION** : 
# 
# In Logistic Regression, the Accuracy score is 95%, which is quite satisfying as this is the first model we have applied, for our data frame.
# 
# 
# ---
# 
# 

# **Decision Tree** : 
# 
# A Decision Tree is a flowchart-like structure in which each internal node represents a test on a feature (e.g. whether a coin flip comes up heads or tails) , each leaf node represents a class label (decision taken after computing all features) and branches represent conjunctions of features that lead to those class labels. The paths from root to leaf represent classification rules.
# 

# In[ ]:


# fit classifier_2
classifier_2.fit(np_train_x, np_train_y.ravel())
pred_2_test_x = classifier_2.predict(np_test_x)
pred_2_train_x = classifier_2.predict(np_train_x)
accuracy_score_2_train_x=accuracy_score(np_train_y,pred_2_train_x)
accuracy_score_2_test_x=accuracy_score(np_test_y,pred_2_test_x)

print("Accuracy for train data : {:.4f} ".format(accuracy_score_2_train_x))
print("Accuracy for test data : {:.4f} ".format(accuracy_score_2_test_x))


# Graphical Representation of Decision Tree
# 
# 

# In[ ]:


"""
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(classifier_2,out_file=None,feature_names=["Age","Income","Family","CCAvg","Education","Mortgage_INT",
                                                                          "SecuritiesAccount","CDAccount",
                                                                          "Online","CreditCard"],filled=True,rounded=True)
graph=graphviz.Source(dot_data)
"""


# In[ ]:


"""
graph
"""


# Confusion Matrix for classifier_2

# In[ ]:


print("Confusion Matrix : ")
print(draw_cm(np_test_y,pred_2_test_x))


# In[ ]:


print("Mean Absolute Error : ", metrics.mean_absolute_error(np_test_y,pred_2_test_x))
print("R2 Score : ",metrics.r2_score(np_test_y,pred_2_test_x))


# Classification Report :

# In[ ]:



print(classification_report(np_test_y,pred_2_test_x))


# **OBSERVATION** :
# 
# In Decision Tree, we got the accuracy score of 98% which comes out to be the BEST TILL NOW.
# 
# 
# ---
# 
# 

# **Random Forest** :
# 
# Random Forest is a flexible, easy to use machine learning algorithm that produces, even without hyper-parameter tuning, a great result most of the time. It is also one of the most used algorithms, because of its simplicity and diversity (it can be used for both classification and regression tasks).

# In[ ]:


# fit classifier_3
classifier_3.fit(np_train_x, np_train_y.ravel())
pred_3_test_x = classifier_3.predict(np_test_x)
pred_3_train_x = classifier_3.predict(np_train_x)
accuracy_score_3_train_x=accuracy_score(np_train_y,pred_3_train_x)
accuracy_score_3_test_x=accuracy_score(np_test_y,pred_3_test_x)

print("Accuracy for train data : {:.4f} ".format(accuracy_score_3_train_x))
print("Accuracy for test data : {:.4f} ".format(accuracy_score_3_test_x))


# In[ ]:

from sklearn.ensemble import RandomForestRegressor

randomforest_model = RandomForestClassifier(max_depth=2, random_state=0)
#randomforest_model = RandomForestRegressor(n_estimators=20, random_state=0)

randomforest_model.fit(np_train_x, np_train_y)


# In[ ]:


Importance = pd.DataFrame({'Importance':randomforest_model.feature_importances_*100}, index=train_x.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )


# Confusion Matrix for classifier_3

# In[ ]:


print("Confusion Matrix : ")
print(draw_cm(np_test_y,pred_3_test_x))


# In[ ]:


print("Mean Absolute Error : ", metrics.mean_absolute_error(np_test_y,pred_3_test_x))
print("R2 Score : ",metrics.r2_score(np_test_y,pred_3_test_x))


# Classification Report :

# In[ ]:



print(classification_report(np_test_y,pred_3_test_x))


# **OBSERVATION** :
# 
# In Randon Forest, we have got 99% accuracy score which is BEST TILL NOW, And also, any model can not be accurate as 100% so, this could be our best model,  but still we are going to apply some more model.
# 
# 
# ---
# 
# 

# **Naive Bayes** : 
# 
# Naive Bayes classifiers are a collection of classification algorithms based on Bayes' Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.
# 
# Bayes Theorem is stated as :
# 
# P(class|data) = (P(data|class) + P(class)) / P(data)
# 
# where P(class|data) is the probablity of class given class and provided data

# In[ ]:


naive_model = GaussianNB()
naive_model.fit(np_train_x, np_train_y)

y_pred = naive_model.predict(np_test_x)


# In[ ]:


print("Accuracy for test data : ",metrics.accuracy_score(np_test_y,y_pred))
print("Accuracy for train data : ",metrics.accuracy_score(np_train_y,naive_model.predict(np_train_x)))


# Confusion Matrix for Naive Bays

# In[ ]:


print("Confusion Matrix : ")
print(draw_cm(np_test_y,y_pred))


# In[ ]:


print("Mean Absolute Error : ", metrics.mean_absolute_error(np_test_y,y_pred))
print("R2 Score : ",metrics.r2_score(np_test_y,y_pred))


# Classification Report :

# In[ ]:



print(classification_report(np_test_y,y_pred))


# **OBSERVATION** :
# 
# In Naive Bayes, we have got the score of 91% which is WORSE SCORE TILL NOW, so still our best model is Random Forest
# 
# 
# ---
# 
# 

# **K-Nearest Neighbors** : 
# 
# The KZNearest Neighbors algorithm is a non-parametric method proposed by Thomas Cover used for classification and regression.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors= 21 , weights = 'uniform', metric='euclidean')
knn.fit(np_train_x, np_train_y)    
y_pred = knn.predict(np_test_x)


# In[ ]:


print("Accuracy for test data : ",metrics.accuracy_score(np_test_y,y_pred))
print("Accuracy for train data : ",metrics.accuracy_score(np_train_y,knn.predict(np_train_x)))


# Confusion Matrix for knn

# In[ ]:


print("Confusion Matrix : ")
print(draw_cm(np_test_y,y_pred))


# In[ ]:


print("Mean Absolute Error : ", metrics.mean_absolute_error(np_test_y,y_pred))
print("R2 Score : ",metrics.r2_score(np_test_y,y_pred))


# Classification Report :

# In[ ]:



print(classification_report(np_test_y,y_pred))


# In K-Nearest Neighbors, we got the accuracy score of 95% which is nearly similar to our Logistic Regression Model, but still this is less accurate than Random Forest.
# 
# 
# ---
# 
# 

# **NEURAL NETWORK :** 
# 
# A neural network is trained by adjusting neuron input weights based on the network's performance on example inputs. If the network classifies an image correctly, weights contributing to the correct answer are increased, while other weights are decreased.
# 

# In[ ]:


MLP_cls=MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
MLP_cls.fit(np_train_x,np_train_y)


# In[ ]:


y_pred=MLP_cls.predict(np_test_x)


# In[ ]:


print("Accuracy for test data : ",metrics.accuracy_score(np_test_y,y_pred))
print("Accuracy for train data : ",metrics.accuracy_score(np_train_y,MLP_cls.predict(np_train_x)))


# Confusion Matrix for MLP_cls

# In[ ]:


print("Confusion Matrix : ")
print(draw_cm(np_test_y,y_pred))


# In[ ]:


print("Mean Absolute Error : ", metrics.mean_absolute_error(np_test_y,y_pred))
print("R2 Score : ",metrics.r2_score(np_test_y,y_pred))


# Classification Report :

# In[ ]:



print(classification_report(np_test_y,y_pred))


# **OBSERVATION** :
# 
# In, Neural Network, the accuracy score is 98% which is very good and also nearly similar to out Decission Tree model, but at the END we have Random Forest to be the Best Model for our DATA FRAME.
# 
# 
# ---
# 
# 

# In[ ]:


import pickle
pickle.dump(randomforest_model,open('random_forest_reg.pkl','wb'))
model = pickle.load(open('random_forest_reg.pkl','rb'))


# 
# 
# # **CONCLUSION :**
# 
# ---
# 
# As from first step we had imported some important libraries that we have used for our Data Frame understanding, cleanig the noise from our data, normalise or clean the data and also used about 6 Models to find the accuracy for the coustomers who can take Personal Loan from our Bank, and we have found that Random Forest comes out to be the best model out of 6 model that we have applied with the accuracy score of 99% which is far better than other model and also we have the worse accuracy of 91% that is from our Naive Bayes model. 
# 
# And some important observations that we have found in our model, for the prediction of coustomers who can take Personal Loan from our Bank, i.e. :
# 1. we observed that, there were two column, ID and ZIP code that was not contributing in our model, but ZIP code could contribute, but as we Know that, they are some random picked values so, it could ruin our model. So, we have dropped ID and ZIP code as well.
# 
# 2. we observed that, Experience and Age are 98% coorelated so, we has to drop one out of both, and also we got some negative values from Experience column that we need to fix and after that we decided to drop Experience column, beccause it might create noise in data.
# 
# 3. we also observed that, CCAvg and Income column data was right skewed so we had to Normalise the data, so we have used yeo-johnson method from Power Transformation to Normalise our data, and Mortgage column was also right skewed so we have used Binning method to Normalise our data, because we got too much noisy data but we could not drop it because it was related to the people who have taken Personal Loan before.
# 
# SO, that was the end of data understanding and normalising it. Now, we had to move to the main part of our model that was spliting the data into 70:30 ratio 70% data to the train set and 30% data to test set, and after that its time to apply some models to predict which coustomers of our bank can take Personal Loan, so we used 6 models to find the same i.e.
# 1. Logistic Regression - Accuracy Score - 95%
# 2. Decision Tree - Accuracy Score - 98%
# 3. Random Forest - Accuracy Score - 99%
# 4. Naives Bayes - Accuracy Score - 91%
# 5. K-Nearest neighbour - Accuracy Score -95%
# 6. Neural Network - Accuracy Score - 98%
# 
# and, the best model was Random Forest with accuracy score of 99% and also, Decision Tree and Neural Network was very good and very close to the Random Forest model.
# Further more details of our models is given below.
# 
# 
# 
# ---
# 
# **-->Details of all our model applied above for Prediction of persons who will take Personal Loan from our Bank.**
# 
# 
# **1. Logistic Regression**
# 
# Accuracy for train data : 0.9569 
# 
# Accuracy for test data : 0.9547
# 
# Mean Absolute Error : 0.04533333333333334 
# 
# R2 Score : 0.477630285152409
# 
# Consusion Matrix : 
# 
# [[1338  , 18]
# 
#  [  50  , 94]]
# 
#  Classification Report :
# 
#               precision    recall  f1-score   support
# 
#            0       0.96      0.99      0.98      1356
#            1       0.84      0.65      0.73       144
# 
#     accuracy                           0.95      1500
# 
#    macro avg    -   0.90     , 0.82  ,    0.85  ,    1500
# 
# weighted avg    -   0.95    ,  0.95   ,   0.95  ,    1500
# 
# ---
# 
# 
# 
# 
# **2. Decision Tree**
# 
# Accuracy for train data : 0.9957 
# 
# Accuracy for test data : 0.9813 
# 
# Mean Absolute Error : 0.018666666666666668 
# 
# R2 Score : 0.7849065880039331
# 
# Confusion Matrix : 
# 
# [[1342  , 14]
# 
#  [  14 , 130]]
# 
# Classification Report :
# 
#               precision    recall  f1-score   support
# 
#            0       0.99      0.99      0.99      1356
#            1       0.90      0.90      0.90       144
# 
#     accuracy                           0.98      1500
# 
#    macro avg    -   0.95   ,   0.95  ,    0.95    ,  1500
# 
# weighted avg    -   0.98      0.98   ,   0.98   ,   1500
# 
# 
# 
# ---
# 
# 
# 
# **3. Random Forest**
# 
# Accuracy for train data : 0.9949 
# 
# Accuracy for test data : 0.9873
# 
# Mean Absolute Error : 0.012666666666666666 
# 
# R2 Score : 0.8540437561455261
# 
# Confusion Matrix : 
# 
# [[1354   , 2]
# 
#  [  17  ,127]]
# 
#  Classification Report :
# 
#               precision    recall  f1-score   support
# 
#            0       0.99      1.00      0.99      1356
#            1       0.98      0.88      0.93       144
# 
#     accuracy                           0.99      1500
#     
#    macro avg    -   0.99    ,  0.94  ,    0.96  ,    1500
# 
# weighted avg    -   0.99  ,    0.99  ,    0.99   ,   1500
# 
# 
# ---
# 
# 
# 
# 
# **4. Naive Bayes**
# 
# Accuracy for test data : 0.9133333333333333 
# 
# Accuracy for train data : 0.9085714285714286
# 
# Mean Absolute Error : 0.08666666666666667 
# 
# R2 Score : 0.0013520157325466187
# 
# Confusion Matrix : 
# 
# [[1293  , 63]
# 
#  [  67  , 77]]
# 
#  Classification Report :
# 
# 
#               precision    recall  f1-score   support
# 
#            0       0.95      0.95      0.95      1356
#            1       0.55      0.53      0.54       144
# 
#     accuracy                           0.91      1500
# 
#    macro avg    -   0.75   ,   0.74    ,  0.75   ,   1500
# 
# weighted avg     -  0.91   ,   0.91    ,  0.91  ,    1500
# 
# ---
# 
# **5. K-Nearest Neighbors**
# 
# Accuracy for test data : 0.946
# 
# Accuracy for train data : 0.9445714285714286
# 
# Mean Absolute Error : 0.054 
# 
# R2 Score : 0.3777654867256637
# 
# Confusion Matrix : 
# 
# [[1351   , 5]
# 
#  [  76  , 68]]
# 
# Classification Report :
# 
#               precision    recall  f1-score   support
# 
#            0       0.95      1.00      0.97      1356
#            1       0.93      0.47      0.63       144
# 
#     accuracy                           0.95      1500
# 
#    macro avg    -   0.94   ,   0.73  ,    0.80   ,   1500
# 
# weighted avg     -  0.95   ,   0.95   ,   0.94   ,   1500
# 
# 
# ---
# 
# **6. NEURAL NETWORK**
# 
# Accuracy for test data : 0.98 
# 
# Accuracy for train data : 0.99
# 
# Mean Absolute Error : 0.02 
# 
# R2 Score : 0.7695427728613569
# 
# Confusion Matrix : 
# 
# [[1346  , 10]
# 
#  [  17  ,127]]
# 
#  Classification Report :
#  
# 
#               precision    recall  f1-score   support
# 
#            0       0.99      0.99      0.99      1356
#            1       0.90      0.90      0.90       144
# 
#     accuracy                           0.98      1500
# 
#    macro avg    -   0.94   ,   0.94   ,   0.94  ,    1500
# 
# weighted avg     -  0.98   ,   0.98   ,   0.98  ,    1500
# 
# ---
# 
# 
# 
