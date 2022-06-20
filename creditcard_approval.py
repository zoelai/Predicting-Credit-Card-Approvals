# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:46:50 2022
@author: Joyun
Data source: http://archive.ics.uci.edu/ml/datasets/credit+approval
"""

# credit card approval system

###########################
#### Import Libraries #####
###########################

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


############################
###### Import Data #########
############################

application = pd.read_csv('datasets/cc_approvals.data')

################
### EDA  ######
###############

# inspect the shape of the dataset
print(application.shape) # 689 rows and 16  columns
print('---------------------------------------------------------')
# inspect data types of each column
application.info()
print('---------------------------------------------------------')
# summary statistics
print(application.describe())
print('---------------------------------------------------------')
# inspect dataframe
print(application.sample(5))

application['+'].value_counts() # balanced data



############################################
###### Check for Multicollinearity #########
############################################

# create dummy variables for categorical variables
dummified_application = pd.get_dummies(application, drop_first = True)
# setting drop_first = True: one hot encode, reduces the correlation between new columns

# plot the correlation matrix to check if multicollinearity exists
corr_matrix = dummified_application.corr()

# drop 0.1 as it is correlated
application = application.drop(['0.1'], axis = 1)
dummified_application = pd.get_dummies(application, drop_first = True)
corr_matrix = dummified_application.corr() # the new corr_matrix looks fine

############################
###### Split  Data #########
############################
train, test = train_test_split(application, test_size = 0.2, random_state = 10)
# We need to split before preprocessing to prevent data leakage


############################################
###### Dealing with missing values #########
############################################

# deal with missing values in the train dataset
train = train.replace('?', np.nan)

# dataframe that include nan
train_columns_null_boolean = train.isnull().any()
train_columns_null = train.columns[train_columns_null_boolean].tolist()
filter_nan_train = train[train_columns_null]

# plot distribution of columns
for i in range(filter_nan_train.shape[1]):
    if i == 1:
        filter_nan_train.iloc[:,i] = filter_nan_train.iloc[:,i].astype('float')
        plt.hist(filter_nan_train.iloc[:,i]) # skewed
        # impute the median value for nan
        filter_nan_train.iloc[:,i].replace(np.nan, filter_nan_train.iloc[:,i].median(), inplace = True)
    else:
        max_val_count = filter_nan_train.iloc[:,i].value_counts()
        # impute mode for categorical columns
        filter_nan_train.iloc[:,i].replace(np.nan, max_val_count.index[0], inplace = True)
        

# place processed value within the train dataframe
#　drop columns with na within the original dataset
train = train.dropna(axis=1)
# add the processed na
train = pd.concat([train, filter_nan_train.reindex(train.index)], axis=1)

#############################################################################

# deal with missing values in the test dataset
test = test.replace('?', np.nan)
# test dataframe that include nan
test_columns_null_boolean = test.isnull().any()
test_columns_null = test.columns[test_columns_null_boolean].tolist()
filter_nan_test = test[test_columns_null]

# plot distribution of columns
for i in range(filter_nan_test.shape[1]):
    if i == 1:
        filter_nan_test.iloc[:,i] = filter_nan_test.iloc[:,i].astype('float')
        plt.hist(filter_nan_test.iloc[:,i]) # skewd
        # plug in the median value for nan
        filter_nan_test.iloc[:,i].replace(np.nan, filter_nan_test.iloc[:,i].median(), inplace = True)
    else:
        max_val_count = filter_nan_test.iloc[:,i].value_counts()
        # plug in mode for categorical columns
        filter_nan_test.iloc[:,i].replace(np.nan, max_val_count.index[0], inplace = True)
        

# place processed value within the test dataframe
#　drop columns with na within the original dataset
test = test.dropna(axis=1)
# add the processed na
test = pd.concat([test, filter_nan_test.reindex(test.index)], axis=1)

# count nans
print(train.isna().sum())
print(test.isna().sum())
# no nans!

############################################
###### Encode Categorical Variables ########
############################################
# encode categorical columns to acheive faster computation, 
# some models require strictly numeric columns only

# assign features and labels for train and test set
x_train = train.drop(['+'], axis = 1)
x_test = test.drop(['+'], axis = 1)
y_test = test.loc[:,['+']]
y_train = train.loc[:,['+']]


application_concat = pd.get_dummies(pd.concat([x_train , x_test])) # to prevent different number of columns
x_train = application_concat[application_concat.index.isin(train.index)]
x_test = application_concat.drop(train.index)

########################
###### Scaling #########
########################

scaler = MinMaxScaler(feature_range=(0,1))
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

######################
### Logistic Reg  ####
######################
logreg = LogisticRegression() # instantiate
logreg.fit(scaled_x_train, y_train)
logreg_pred = logreg.predict(scaled_x_test)


print(confusion_matrix(y_test, logreg_pred))
print("Accuracy: ", logreg.score(scaled_x_test, y_test)) # 0.884

##################################
## Optimizing Model Performance ##
##################################

# Define the grid of values for tol and max_iter
tol = [0.1, 0.01, 0.001, 0.0001]
max_iter = [50, 100, 150, 200]

# Create a dictionary where tol and max_iter are keys 
param_grid = dict(tol=tol, max_iter=max_iter)

# Instantiate GridSearchCV
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Fit grid_model to the data
grid_model_result = grid_model.fit(scaled_x_train, y_train)

# Summarize results
print('======================================================================')
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))

# Extract the best model and evaluate it on the test set
best_model = grid_model_result.best_estimator_
print("Accuracy of logistic regression classifier: ", best_model.score(scaled_x_test, y_test))
print('======================================================================')








