    #importing libraries
#importing the libraries for data analysis
#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Read the data and Assign price to y variable and rest to x
db=pd.read_csv(r'C:\Users\prkavis\Music\zshoes\product_data.csv')
X=db.drop(['price'],axis=1)
Y=db['price']

#Do a train test split

x_train,x_test,y_train,y_test=train_test_split(X,Y, train_size=0.8, random_state=42)

print("Training model")
#Train a randomforestregressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
#Print r2score
from sklearn import metrics
metrics.r2_score(y_test, y_pred)

#saving my model
pickle.dump(regressor, open('model.pkl','wb'))
print("dumping complete")
#
# #loading the model






