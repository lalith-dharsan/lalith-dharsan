#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[4]:


df = pd.read_csv("Downloads\Copy of loan - loan.csv")


# In[5]:


df.head(0)


# In[6]:


df['LoanAmount_log']= np.log(df.LoanAmount)
df['LoanAmount_log'].hist(bins=20)


# In[7]:


df['total'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['total_log'] = np.log(df['total'])
df['total_log'].hist(bins=20)


# In[24]:


df.head(2)


# In[9]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace =True)


# In[25]:


df['Dependents'].fillna(df['Dependents'].mode()[0], inplace= True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)
df['LoanAmount_log'] = df['LoanAmount_log'].fillna(df['LoanAmount_log'].mean())


# In[26]:


df.isnull().sum()


# In[27]:


df.Credit_History


# In[31]:


x= df.iloc[:, np.r_[1:5, 9:11,13:15]].values
y= df.iloc[:,12].values
x


# In[41]:


y


# In[33]:


df.head(0)


# In[38]:


df.Gender.value_counts()
sns.countplot(x='Gender', data =df, legend =False )


# In[68]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# from sklearn.preprocessing import LabelEncoder as label
# label_x = label()
# label_y = label()

# In[176]:


for i in range(0,5):
    X_train[:,i] = label_x.fit_transform(X_train[:,i])
    X_train[:,7] = label_x.fit_transform(X_train[:,7])   
Y_train = label_y.fit_transform(Y_train)
X_train


# In[177]:


Y_train 


# In[178]:


for i in range(0,5):
    X_test[:,i] = label_x.fit_transform(X_test[:,i])
    X_test[:,7] = label_x.fit_transform(X_test[:,7])   
Y_test = label_y.fit_transform(Y_test)
Y_test


# In[179]:


from sklearn.preprocessing import StandardScaler as std
ss = std()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
X_pred


# In[180]:


from sklearn.ensemble import RandomForestClassifier as rclf
r_clf = rclf()
r_clf.fit(X_train,Y_train)


# In[181]:


from sklearn import metrics
Y_pred = r_clf.predict(X_test)
print("acc of random forest clf is", metrics.accuracy_score(Y_pred, Y_test))
Y_pred


# In[182]:


from sklearn.naive_bayes import GaussianNB as gas
nb_clf = gas()
nb_clf.fit(X_train,Y_train)


# In[183]:


Y_pred1 = nb_clf.predict(X_test)
print("acc of GaussianNB clf is",metrics.accuracy_score(Y_pred1,Y_test))
Y_pred1


# In[187]:


from sklearn.tree import DecisionTreeClassifier as dest
dt_clf = dest()
dt_clf.fit(X_train,Y_train)


# In[188]:


Y_pred2 = dt_clf.predict(X_test)
print("acc of DecisionTreeClassifier clf is",metrics.accuracy_score(Y_pred2,Y_test))
Y_pred2


# In[189]:


from sklearn.neighbors import KNeighborsClassifier as kne
kn_clf = kne()
kn_clf.fit(X_train,Y_train)


# In[190]:


Y_pred3 = kn_clf.predict(X_test)
print("acc of DecisionTreeClassifier clf is",metrics.accuracy_score(Y_pred3,Y_test))
Y_pred3


# In[ ]:





# In[ ]:




