#regression exercise
#find best fit line
#model data
#linear regression

#regression for stock prices

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL',api_key = 'g3yXV_Q8k4zrFdiXxb_x')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


forcast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forcast_out = int(math.ceil(.01*len(df)))#number of days out
#predict 10% of data frame
#using data from 10 days ago to predict today

df['label']=df[forcast_col].shift(-forcast_out)#shift columns negatively
#each label column will now be the adjusted close 10 days into the future
df.dropna(inplace = True)#drop rows in data frame that have NaN values
print(df.head())

#data set is created now we can train



X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X)#scale x before feeding to classifier

#remember when processing new data, must scale it along side all data

print(len(X),len(y))

#create traiing set

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = .2)

#fit training data to model
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train,y_train)

#test model on test data
accuracy = clf.score(X_test, y_test)


print(accuracy)





