#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING ALL LIBRARIES & PACKAGES


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score


# In[ ]:





# In[3]:


#READING DATASET


# In[ ]:





# In[4]:


data=pd.read_csv("heart_disease_dataset.csv")


# In[5]:


data.shape


# In[6]:


data


# In[7]:


#checking if any value is containing null
data.info()


# In[8]:


# Number of female = 138, Number of male = 165
data.groupby("target").size()


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


#Data exploration of all the features and printing as a histogram
data.hist(figsize=(15,15))
plt.show()


# In[ ]:





# In[ ]:





# In[10]:


#Almost 70% female has heart disease & 45% male has heart disease
sns.barplot(data["sex"],data["target"])


# In[ ]:





# In[ ]:





# In[11]:


#Target is showing according to sex & age
sns.barplot(data["sex"],data["age"],hue=data["target"])
plt.show()


# In[ ]:





# In[ ]:





# In[12]:


#creating four displots
plt.figure(figsize=(15,13))

plt.subplot(221)
sns.distplot(data[data['target']==0].age)
plt.title('Age of patients without heart disease')

plt.subplot(222)
sns.distplot(data[data['target']==1].age)
plt.title('Age of patients with heart disease')

plt.subplot(223)
sns.distplot(data[data['target']==0].thalach )
plt.title('Max heart rate of patients without heart disease')

plt.subplot(224)
sns.distplot(data[data['target']==1].thalach )
plt.title('Max heart rate of patients with heart disease')

plt.show()


# In[ ]:





# In[ ]:





# In[13]:


#DATA PREPROCESSING


# In[14]:


X,y=data.loc[:,:"thal"],data["target"]


# In[15]:


X


# In[16]:


y


# In[ ]:





# In[ ]:





# In[17]:


#SPLITTING THE DATA SET INTO TRAIN & TEST


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[22]:


y


# In[23]:


y_train.shape


# In[24]:


y_test.shape


# In[ ]:





# In[ ]:





# In[25]:


#IMPLEMENTING USING DECISION TREE CLASSIFIER ALGORITHM


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=10)
dt.fit(X_train,y_train)


# In[27]:


X_test


# In[ ]:





# In[28]:


prediction=dt.predict(X_test)


# In[29]:


prediction


# In[30]:


accuracy_dt=accuracy_score(y_test,prediction)*100


# In[31]:


accuracy_dt


# In[ ]:





# In[32]:


category=['No the patient is not having heart disease',
          'Yes the patient is having heart disease please consult with a heart specialist']


# In[ ]:





# In[33]:


#CHECKING WITH CUSTOM DATA FOR DECISION TREE CLASSIFIER


# In[34]:


custom_data=np.array([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])


# In[35]:


custom_data_prediction_dt=dt.predict(custom_data)


# In[36]:


int(custom_data_prediction_dt)


# In[37]:


print(category[int(custom_data_prediction_dt)])


# In[ ]:





# In[ ]:





# In[38]:


#IMPLEMENTING USING KNN ALGORITHM


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)


# In[40]:


prediction_knn=knn.predict(X_test)


# In[41]:


accuracy_knn=accuracy_score(y_test,prediction_knn)*100


# In[42]:


accuracy_knn


# In[ ]:





# In[ ]:





# In[43]:


#CHECKING WITH CUSTOM DATA FOR KNN ALGORITHM


# In[44]:


custom_data=np.array([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])


# In[45]:


custom_data_prediction_knn=knn.predict(custom_data)


# In[46]:


int(custom_data_prediction_knn)


# In[47]:


print(category[int(custom_data_prediction_knn)])


# In[ ]:





# In[ ]:





# In[48]:


#COMPARING BOTH ALGORITHMS


# In[49]:


algorithms=['Decision Tree','KNN']
scores=[accuracy_dt,accuracy_knn]


# In[50]:


sns.barplot(algorithms,scores)
plt.show()


# In[ ]:





# In[51]:


#END


# In[ ]:




