#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

import plotly as plot
import plotly.express as px
import plotly.graph_objs as go

import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot


# In[2]:


data=pd.read_csv("heart_disease_dataset.csv")


# In[3]:


data.shape


# In[4]:


data


# In[5]:


data.info()


# In[6]:


y=data["target"]


# In[7]:


data.groupby("target").size()


# In[8]:


data.size


# In[9]:


data.describe()


# In[10]:


data.info()


# In[ ]:





# In[ ]:





# In[11]:


data.hist(figsize=(10,10))
plt.show()


# In[ ]:





# In[ ]:





# In[12]:


sns.barplot(data["sex"],data["target"])


# In[ ]:





# In[ ]:





# In[13]:


sns.barplot(data["sex"],data["age"],hue=data["target"])
plt.show()


# In[ ]:





# In[ ]:





# In[14]:


numeric_coloumns=["trestbps","chol","age","oldpeak","thalach"]


# In[15]:


sns.heatmap(data[numeric_coloumns].corr(),annot=True,cmap='terrain',linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[ ]:





# In[ ]:





# In[16]:


plt.figure(figsize=(10,10))
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





# In[17]:


#DATA PREPROCESSING


# In[18]:


X,y=data.loc[:,:"thal"],data["target"]


# In[19]:


X


# In[20]:


y


# In[ ]:





# In[ ]:





# In[131]:


#DATA PREPROCESSING FOR KNN ONLY 

#from sklearn.preprocessing import StandardScaler

#std=StandardScaler().fit(X)
#X_std=std.transform(X)


# In[132]:


#X_std


# In[ ]:





# In[ ]:





# In[21]:


#SPLITTING THE DATA SET INTO TRAIN & TEST


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


y


# In[28]:


y_train.shape


# In[29]:


y_test.shape


# In[ ]:





# In[ ]:





# In[ ]:


#IMPLEMENTING USING DECISION TREE CLASSIFIER ALGORITHM


# In[53]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=10)
dt.fit(X_train,y_train)


# In[54]:


X_test


# In[ ]:





# In[ ]:





# In[154]:


prediction=dt.predict(X_test)


# In[155]:


prediction


# In[156]:


accuracy_dt=accuracy_score(y_test,prediction)*100


# In[157]:


accuracy_dt


# In[ ]:





# In[59]:


dt.feature_importances_


# In[73]:


def plot_feature_importances(model):
    plt.figure(figsize=(8,6))
    n_features = 13
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances(dt)
plt.savefig('feature_importance')


# In[ ]:





# In[ ]:





# In[ ]:


#IMPLEMENTING USING DECISION TREE CLASSIFIER


# In[74]:


data


# In[84]:


category=['No the patient is not having heart disease','Yes the patient is having heart disease please consult with a heart specialist']


# In[ ]:





# In[ ]:





# In[93]:


#CHECKING WITH CUSTOM DATA


# In[94]:


custom_data=np.array([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])


# In[95]:


custom_data_prediction_dt=dt.predict(custom_data)


# In[96]:


int(custom_data_prediction_dt)


# In[97]:


print(category[int(custom_data_prediction_dt)])


# In[ ]:





# In[ ]:





# In[ ]:


#IMPLEMENTING USING KNN ALGORITHM


# In[133]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)


# In[134]:


prediction_knn=knn.predict(X_test)


# In[135]:


accuracy_knn=accuracy_score(y_test,prediction_knn)*100


# In[136]:


accuracy_knn


# In[ ]:





# In[ ]:





# In[ ]:


#CHECKING WITH CUSTOM DATA


# In[125]:


custom_data=np.array([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])


# In[126]:


custom_data_prediction_knn=knn.predict(custom_data)


# In[127]:


int(custom_data_prediction_knn)


# In[128]:


print(category[int(custom_data_prediction_knn)])


# In[ ]:





# In[ ]:





# In[ ]:


#BEST RANGE OF KNN


# In[142]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    prediction_knn=knn.predict(X_test)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))


# In[143]:


k_scores


# In[144]:


scores_list


# In[148]:


plt.plot(k_range,scores_list)


# In[ ]:





# In[ ]:





# In[ ]:


#COMPARING BOTH ALGORITHMS


# In[159]:


algorithms=['Decision Tree','KNN']
scores=[accuracy_dt,accuracy_knn]


# In[161]:


sns.barplot(algorithms,scores)
plt.show()


# In[ ]:





# In[162]:


#END


# In[ ]:




