#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)


# In[42]:


# read data
df = pd.read_csv('../data/preprocessed_models_data.csv')
df.describe()

# Labels are the values we want to predict
labels = np.array(df['SSL SCORE'])
df = df.drop('SSL SCORE', axis = 1)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
df = np.array(df)


# In[45]:


train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25, random_state = 42)


# # Random Forest

# In[48]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels);


# In[49]:


predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[50]:


mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

