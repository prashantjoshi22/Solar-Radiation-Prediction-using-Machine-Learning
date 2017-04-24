
# coding: utf-8

# ## Preprocessing the Data

# In[2]:

import pandas as pd
import numpy as np
from pandas import Series
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[3]:

data = pd.read_csv('/home/prashant/Desktop/solardata.csv',skiprows = 2)


# In[4]:

#Changing the columns name because of the space problems
col_names =data.columns
col_names = data.columns.str.replace(' ','_')
print(col_names)


# In[5]:

#Dropping the empty columms which are of no use

data.columns = col_names
drop_list = []

for col in col_names:
   
    if(data[col].isnull().sum() > data.shape[0]/2):
        drop_list.append(col)
        
data.drop(drop_list,axis =1,inplace =True)
data.head()       


# In[6]:

standard_area =5.7
efficiency = 0.20
data['Power_Produced'] = data['GHI'] * standard_area * efficiency
data.drop(['GHI','Snow_Depth'],axis =1,inplace= True)


# In[7]:

#Creating the new variable with name as Date_Time and this will contian all the information regarding date,year,and time

data['Date_Time'] = data.Year.astype(str) + "/" + data.Month.astype(str) + "/" + data.Day.astype(str) + " " + data.Hour.astype(str)+ ":" + data.Minute.astype(str)    
data['Date_Time']=pd.to_datetime(data['Date_Time'])

#Printing year,month,day and time of first 6 rows
print(data.Date_Time.head())


# In[8]:

plt.plot(data.Date_Time.dt.hour,data.Power_Produced)
plt.title('Plot between Different hours of the day and Power Produced')
plt.xlabel('Hours')
plt.ylabel('Power Produced')



# In[9]:

#Droping the column [Year,Month,Day,Hour,Minute] because there information is covered in columnn 'Date_Time'
data.drop(['Year','Month','Day','Hour','Minute'],axis =1,inplace=True)
data.drop('Fill_Flag',axis =1,inplace=True)


# In[10]:

#Checking that is there any missing values is present or not in the data
print("The number of missing values in the column :-")
data.isnull().sum()


# In[11]:

data['DHI'].isnull()
col_name = data.columns
data['Pressure'].values



# In[12]:

data.fillna(data.mean(),inplace = True)
print("All missing values are replaced by the mean of each feature")            


# In[13]:

data.info()


# In[14]:

#Create a new Dataframae
time = data.Date_Time.dt.time
ts = data.loc[: ,['Date_Time','Power_Produced']]
ts =ts.set_index('Date_Time')


# In[15]:

plt.plot(data.Date_Time.dt.time,data.Power_Produced)
plt.title('Plot between time of the day and PowerProduced at that time')
plt.xlabel('Months')
plt.ylabel('PowerProduced')
plt.show()


# In[16]:

plt.plot(data.Date_Time.dt.day,data.Power_Produced)
plt.title('Plot between every day in week and PowerProduced')
plt.xlabel('Months')
plt.ylabel('PowerProduced')
plt.show()


# In[17]:

plt.plot(data.Date_Time.dt.month,data.Power_Produced)
plt.title('Plot between months and PowerProduced')
plt.xlabel('Months')
plt.ylabel('PowerProduced')
plt.show()


# In[18]:

#print(data.head())
#plt.matshow(data.corr())
import matplotlib.pylab as plt1
import seaborn as sns
corr = data.corr()

fig, (axis1) = plt1.subplots(1,1,figsize=(15,5))
sns.heatmap(corr,annot =True ,linewidths=2,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
print("Corelation Matrix :")
plt1.show()


# In[19]:


#Sepertating the target and response variables
X = data.loc[:,data.columns != 'Power_Produced']
X = X.loc [ : , X.columns != 'Clearsky_GHI']
X = X.loc[: ,X.columns != 'Date_Time']
X = X.drop(['DNI','DHI','Clearsky_DHI','Clearsky_DNI'],axis =1)
y = data.Power_Produced
y= pd.DataFrame(y)
y['Power_Produced'] =y


# In[20]:

X.to_csv("X_modified.csv",index=False)
y.to_csv("y_modified.csv",index=False)


# In[ ]:



