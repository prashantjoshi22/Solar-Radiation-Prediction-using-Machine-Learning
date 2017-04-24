
# coding: utf-8

# ## Applying Linear Regression Model

# In[ ]:

import statsmodels.api as sm
import pandas as pd
import numpy as np
from pandas import Series
import matplotlib.pylab as plt
from sklearn import metrics
#%matplotlib inline
from  sklearn.cross_validation import train_test_split
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[189]:

data_X = pd.read_csv('X_modified.csv')
data_y = pd.read_csv('y_modified.csv')


# In[190]:

#Spliting the dataset into training and testing set

X_train,X_test,y_train,y_test = train_test_split(data_X,data_y,random_state =0)


# In[191]:

model = sm.OLS(y_train, X_train)
fit=model.fit()
print(fit.summary())


# In[192]:

y_pred =(fit.predict(X_test))


# In[193]:

y_test_list = y_test['Power_Produced'].tolist()


# ### So the  linear regression model is not a good model 

# In[194]:

print("Length of testing data")
print(len(y_pred))
print(len(y_test))
pred_val = []
actual_val =[]
for i in range (0 , 100):
    pred_val.append(y_pred[i])
    actual_val.append(y_test_list[i])


# In[195]:

pred_series = pd.Series(pred_val)
actual_series = pd.Series(actual_val)


# In[196]:

data_plot = pd.DataFrame(pred_val,actual_val)
print(data_plot.head())
data_plot=data_plot.reset_index()
data_plot.head()
col_name = ['Actual_Value','Predicted_Value']
data_plot.columns = col_name
#data_new.plot(stacked=False)


# In[197]:

data_plot.plot()
plt.title('Comparasion between Predicted Values and Actual Values of the Power Produced')
plt.ylabel('PowerProduced')
plt.show()


# ## Poor model

# In[198]:

#Printing RMSE value
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

