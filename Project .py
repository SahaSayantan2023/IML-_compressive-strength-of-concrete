#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats


# In[5]:


pip install xgboost


# In[6]:


concrete = pd.read_excel("C:\\Users\\sayan\\OneDrive\\Documents\\Concrete data.xlsx")


# In[7]:


concrete.head()


# In[10]:


concrete.columns = ['cement','blastfurnace','flyAsh','water','superplasticizer','courseAggregate','fineaggregate','age','strength']


# In[11]:


concrete.head()


# In[13]:


concrete.shape


# In[14]:


concrete.isnull().sum()


# In[15]:


concrete.duplicated().sum()


# In[16]:


concrete.info()


# In[17]:


concrete.describe()


# In[18]:


corr = concrete.corr()
corr


# In[13]:


sns.heatmap(corr, annot=True, cbar=True, cmap='coolwarm')


# In[19]:


X = concrete.drop('strength',axis=1)
y = concrete['strength']


# In[20]:


X.shape


# In[21]:


y.shape


# In[22]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# In[25]:


for col in X_train.columns:
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
    plt.show()
    
    
    


# In[26]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()


# In[45]:


X_train_transformed = pt.fit_transform(X_train)
X_test_transformed  = pt.transform(X_test)


# In[46]:


X_train_transformed = pd.DataFrame(X_train_transformed, columns=X_train.columns)

for col in X_train_transformed.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
    
    
    plt.subplot(122)
    sns.distplot(X_train_transformed[col])
    plt.title(col)
    plt.show()


# In[47]:


from sklearn.preprocessing import StandardScaler
scalr = StandardScaler()


# In[48]:


X_train_transformed = scalr.fit_transform(X_train_transformed)
X_test_transformed = scalr.transform(X_test_transformed)


# In[53]:


X_train_transformed


# In[54]:


from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from sklearn.metrics import mean_squared_error,r2_score


# In[57]:


models = {
    "Ir":LinearRegression(),
    "Iss":Lasso(),
    "Ridg":Ridge(),
    "dtr":DecisionTreeRegressor(),
    'xgb': XGBRFRegressor()
}

for name, md in models.items():
    md.fit(X_train_transformed,y_train)
    y_pred = md.predict(X_test_transformed)
    
    print(f"{name} : mse: {mean_squared_error(y_test,y_pred)}, r2 score : {r2_score(y_test,y_pred)}")


# In[59]:


xgb = XGBRFRegressor()
xgb.fit(X_train_transformed,y_train)
y_pred = xgb.predict(X_test_transformed)
r2_score(y_test,y_pred)    


# In[60]:


import pickle
pickle.dump(xgb,open('model.pk1','wb'))


# In[75]:


def pred_strength(cem,blastf,flyas,water,superplaster,courseagg,fineagg,age):
    features = np.array([[cem,blastf,flyas,water,superplaster,courseagg,fineagg,age]])

    pred = xgb.predict(features).reshape(1,-1)

    return pred[0]


# In[82]:


cem = 158.60
blastf = 148.90
flyas = 116.00
water = 175.10
superplaster = 15.00
courseagg = 953.3
fineagg = 719.70
age = 28

strength = pred_strength(cem,blastf,flyas,water,superplaster,courseagg,fineagg,age)


# In[83]:


print(strength)


# In[ ]:





# In[ ]:




