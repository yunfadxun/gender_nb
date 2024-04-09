#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[4]:


df= pd.read_csv("sext.csv")
df.info()


# In[5]:


for i in df.columns[0:2]:
    plt.figure()
    sns.boxplot(df[i],hue=df[i])


# In[29]:


for i in df.columns[0:2]:
    Q1= df[i].quantile(0.25)
    Q3= df[i].quantile(0.75)
    IQR=Q3-Q1
    upper= Q3+1.5*IQR
    lower= Q1-1.5*IQR
    df=df[(df[i]>=lower)&(df[i]<=upper)]
    
    plt.figure()
    sns.boxplot(df[i], hue=df[i])


# In[30]:


df.shape


# In[31]:


df["Sex"]=df["Sex"].replace({"Female":0,"Male":1})
sns.countplot(df["Sex"],data=df)


# In[32]:


#defining x and y
x= df.drop("Sex",axis=1)
y= df["Sex"].values.reshape(-1,1)


# In[33]:


#data train split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,random_state=44, train_size=0.7)


# In[34]:


#scaling
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.fit_transform(x_test)


# In[35]:


#naive bayes
from sklearn.naive_bayes import GaussianNB
basemodel= GaussianNB()
basemodel.fit(x_train,y_train)
y_base= basemodel.predict(x_test)


# In[36]:


#accuracy score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
print(classification_report(y_test,y_base))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_base)).plot()


# In[38]:


#hyperparameter tuning
params={"var_smoothing": np.logspace(0,-9,num=100)}


# In[40]:


#model with tuning
from sklearn.model_selection import RandomizedSearchCV
rs= RandomizedSearchCV(GaussianNB(), params, cv=5, n_jobs=-1, n_iter=100)
rs.fit(x_train,y_train)


# In[41]:


rs.best_params_


# In[46]:


good_model= rs.best_estimator_
y_pred= good_model.predict(x_test)
print(classification_report(y_test,y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred)).plot()


# In[ ]:




