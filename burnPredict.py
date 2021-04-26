#!/usr/bin/env python
# coding: utf-8

# # Using Surveyed Data To Assume Employee Burn Out

# ## Import Statements And Setup 

# In[53]:


import numpy as np
import pandas as pd
import dsx_core_utils
import matplotlib.pyplot as plt
from dsx_core_utils import ProjectContext
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
# Add asset from file system
pc = ProjectContext.ProjectContext('Datathon-21', 'burnPredict.ipynb', 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InNwa2M2YmIiLCJyb2xlIjoibWxhZG0iLCJ1aWQiOiIxMDM2IiwiZGlzcGxheV9uYW1lIjoiU1BLQzZCQiIsImlhdCI6MTYxOTM2MTExOCwiZXhwIjoxNjE5NDA3OTE4fQ.A3g096CgOvs5jKXP0X2Fk7QgtJP10R7_Dozjv-oTqAp7Wm7dN8vWGfGTXferxvpKjAHY34wqHRd-1cqvMEutGpZ2sYoAtmPfOreYOvx8EZOJQDFBIRzMDRMxHsZs2zFEXmpb5Cw2BmU1fnU1MeHIfRiSEG1jRWrfplJgvUX34qN4pxt2ps_pRc5-YSVb-TVWEftSYM95Qmjt7OqHWS-YqrWxa4g_r1gnr9dpfQ3xyH-07F1zFlJBWQDRGmvAf-qQSNA23Tg3fOu4avqhxYzdNv4IolE7GPDPLVU-mDSoeVkfV0OzSqU8-DpsguHSv5HMv7Xx1d0wnJUNIkB3SGM0oA', '148.100.104.170')
filepath = dsx_core_utils.get_local_dataset(pc, 'train.csv')
tData = pd.read_csv(filepath)
tData.head()


# ## Turning Dataset Into Usable DataFrame

# In[54]:


keys = tData.keys()
tData.dropna()
tDataFrame = pd.DataFrame(tData, columns=keys)
tDataFrame = tDataFrame.dropna()

tDataFrame['MEDV'] = tData.Burn_Rate


# ## Setup For Both Training and Testing Data Sets

# In[55]:


X = pd.DataFrame(np.c_[tDataFrame['Designation'], tDataFrame['Resource_Allocation'],tDataFrame['Mental_Fatigue_Score']], columns=['Designation','Resource_Allocation', 'Mental_Fatigue_Score'])
Y = tDataFrame['MEDV']


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=9)


# ## Initializing and Training The Linear Regression Model

# In[57]:


model = LinearRegression()


# In[61]:


model.fit(X_train, y_train)


# ## Making Predictions Using The Model

# In[62]:


pred = model.predict(X_test)


# ## Determining And Printing Effectiveness Data

# In[63]:


test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

test_set_r2 = r2_score(y_test, pred)


# In[64]:


print("Mean Squared Error (Lower is better): " + str(test_set_rmse))
print("Score (Closer to 1 is better): "+ str(test_set_r2))


# In[65]:


print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')


# # Survey Prototype

# In[66]:


des = float(input("What is your designation from 0-5?\n"))
ra = float(input("How would you rate your resource allocation from 1-10?\n"))
mfs = float(input("From 1-10 how mentally fatigued do you feel?\n"))


# In[67]:


print("Burn Out Rate:" + str(model.predict([des,ra,mfs])[0]))


# In[ ]:




