# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:59:47 2020

@author: siddhant modi
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/siddh/Desktop/Data Science Pratice/DS_Salary_Predictor/eda_data.csv')

# choose relevant columns 
df.columns

df_model = df[['Average_Salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','Job_State','Company Age'
               ,'python_yn','spark','aws','excel','Seniority','desc_len']]


# get dummy data 
df_dum = pd.get_dummies(df_model)


# train test split

X = df_dum.drop('Average_Salary', axis =1)
y = df_dum.Average_Salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=0)

# multiple linear regression 
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring =  'neg_mean_absolute_error', cv= 6))

# lasso regression 
lm_l = Lasso(alpha=1.8)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 6))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/10)
    lml = Lasso(alpha=(i/10))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

# random forest 
rf = RandomForestRegressor()
np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error',cv=6))

# tune models GridsearchCV 

parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=6)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

# test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)


mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)


