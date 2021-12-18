#!/usr/bin/env python
# coding: utf-8

# ### Provident Credit Union Exercise  
# > #### Analyst: Frederick $($Fritz$)$ Zuhl
# > #### December 17, 2021
# #### Part 2 - Create Model 
# + #### Some Additional Feature Preparaton
# + #### Create and Fit XGBoost model
# + #### Review Most Significant Features
# + #### Score All Active Accounts  
# *** 
# ***
# <br>  
# <br>

# In[5]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree, plot_importance
import matplotlib.pyplot as plt


# In[6]:


data_features = pd.read_csv("data_with_features.csv")
#data_features.set_index(data_features['account_id'], inplace=True)


# In[7]:


# Check on data from EDA efforts.
print(data_features.info())


# *** 
# ***
# ### Part 2 / Section 1 - One-Hot encoding.
# + #### Due to the nature of deep learning algorithms, such as XG boost, it is necessary to convert categorical data into numerical.
# + #### Include One-Hot encoded features into analytic dataset.
# ***

# In[8]:


# XGBoost needs categorical values to be one-hot encoded.
# This code section converts the features that are in the form of a category into 1-hot encoding

#loan
fico_bins_ds = pd.get_dummies(data_features['fico_bin'], prefix='FICO')
# ntriggers_bin
ntrigger_bins_df= pd.get_dummies(data_features['ntriggers_bin'], prefix='NTRIGGERS')
one_hot_features = fico_bins_ds.join(ntrigger_bins_df)
# num_times_neg_bin
num_times_neg_bins_df = pd.get_dummies(data_features['num_times_neg_bin'], prefix="NUMtimesNEG")
one_hot_features = one_hot_features.join(num_times_neg_bins_df)
# max_days_neg_bin
max_days_neg_bins_df = pd.get_dummies(data_features['max_days_neg_bin'], prefix="MAXdaysNEG")
one_hot_features = one_hot_features.join(max_days_neg_bins_df)
# ck_returns_bin
ck_returns_bins_df = pd.get_dummies(data_features['ck_returns_bin'], prefix="CTRETURNS")
one_hot_features = one_hot_features.join(ck_returns_bins_df)
# ave_bal6_binned/ave_bal3_binned
aveBal6_bins_df = pd.get_dummies(data_features['avg_bal6_binned'], prefix='AVEBAL6')
one_hot_features = one_hot_features.join(aveBal6_bins_df)
#
aveBal3_bins_df = pd.get_dummies(data_features['avg_bal3_binned'], prefix='AVEBAL3')
one_hot_features = one_hot_features.join(aveBal3_bins_df)
#
print("Shape of one-hot features:", one_hot_features.shape)


# In[9]:


# Join one-hot features to general dataset.
data_features2 = data_features.join(one_hot_features)


# In[10]:


# Check data after 1-hot encoding
print("size of data_features2:", data_features2.shape)
print("Columns in data_features2", data_features2.info())


# In[11]:


# Keep features needed for model
# Drop columns not needed
drop_columns = ['fico_bin', 'ntriggers_bin', 'num_times_neg_bin', 'max_days_neg_bin', 'ck_returns_bin',
                'avg_bal6_binned', 'avg_bal3_binned']
drop_columns2 = ['fico_b', 'ntriggers', 'num_times_neg', 'max_days_neg', 'ck_returns', 'avg_bal6', 'avg_bal3'] + drop_columns
# optional drop
drop_columns3 = ['account_id'] + drop_columns2


# In[12]:


analytic_dataset = data_features2.drop(drop_columns2, axis=1)
print(analytic_dataset.info())


# In[13]:



# inspect recent work with Excel
analytic_dataset.to_csv("analytic_dataset.csv")


# #### The story so far...
# The data from Step 1, EDA and Feature Extraction phase, was read into Python. All of the categorical features had to be converted into one-hot encoding, since XGboost work best with numerical data.
# After the one-hot encoded features were added to the analytical dataset, many of the original fields need to be removed before submitting to XGboost.

# ***
# ***
# ### Part 2 / Section 2 - Divide into training and test data.
# + #### Further preparation for XGBoost Algorithm
# + #### Calculate class imbalance
# + #### Create datasets for RimAge and RimAgeReverse
# ***

# In[14]:


# Split data into charged-off and non charged-off.
# Since the objective of the model is to identify accounts that are most likely to commit fraudulent activities,
# we will select accounts that are active.
active_accounts = analytic_dataset[analytic_dataset['status']=='Active']
data_CO = analytic_dataset[analytic_dataset['cos']==1]


# In[15]:


# Check
print("Active accounts shape:",active_accounts.shape)
print("Charged-off accounts shape:", data_CO.shape)


# In[16]:


# Split 'active' accounts into 20% test and leave the rest to
# score later.
train_active_accts, test_active_accts = train_test_split(active_accounts,
                                         train_size=0.2,
                                         test_size=0.8,
                                         shuffle=True,
                                         #stratify=stratification_fields,
                                         random_state=417)


# In[17]:


train_active_accts.shape


# In[18]:


print(train_active_accts.head())


# In[19]:


# For training set, add the 20%-sample of active accounts with all of the charged-off accounts.
analytic_dataset_model = pd.concat([data_CO, train_active_accts])
analytic_dataset_model.drop(['status','account_id'], axis=1, inplace=True)
print("Shape of analytic dataset:", analytic_dataset_model.shape)
class_imbalance_factor = train_active_accts.shape[0]/data_CO.shape[0]
print("Class Imbalance Factor: ", class_imbalance_factor)


# In[20]:


# calculate heuristic class weighting
# This will place extra penality on COS=1 / charged off accounts.
# Dealing with highly imbalanced data.
# Will be used in XGboost model
from sklearn.utils.class_weight import compute_class_weight

# calculate class weighting according to training data
weighting = compute_class_weight(class_weight='balanced',
                                 classes=[0,1],
                                 y=analytic_dataset_model['cos']
                                 )
print(weighting)


# In[21]:


# Break training data into Independent and Dependent fields.
y_train = analytic_dataset_model['cos'].to_numpy()
x_train = analytic_dataset_model.loc[:, analytic_dataset_model.columns != 'cos']


# In[22]:


# Create 2 training sets.
# I suspect that the rim_age may be incorrectly prepared, with the age actually reversed with the 42-month window.
# In the EDA phase, I created an additional field, rim_age_reverse, that is the mirror reverse of the rim_age field.
x_train_rimAge = x_train.drop(['rim_age_reverse'], axis=1).copy(deep=True)
x_train_rimAgeRev = x_train.drop(['rim_age'], axis=1).copy(deep=True)


# In[23]:


x_train_rimAgeRev.columns


# ***
# ***
# ### Part 2 / Section 3 - Define and Fit Model
# + #### Using both training data sets, use XGBoosting algorithm to create classification model.
# + #### Review the features and their relative importance.
# + #### Score all 'Active' accounts with model output.
# ***

# In[24]:


# define rimAgeRev model - Age of account is reversed within time window.
model_rimAgeRev = XGBClassifier(scale_pos_weight=class_imbalance_factor,
                      objective="binary:logistic",
                      learning_rate=.1,
                      max_depth=9,
                      eval_metric=['logloss'],
                      early_stopping_rounds=20,
                      use_label_encoder=False,
                      subsample=0.5,
                      verbosity=0,
                      n_estimators=150
                      )
# fit model
model_rimAgeRev.fit(x_train_rimAgeRev, y_train)


# In[25]:


# define rimAge model - Age of account as provided.
model_rimAge = XGBClassifier(scale_pos_weight=class_imbalance_factor,
                      objective="binary:logistic",
                      learning_rate=.1,
                      max_depth=9,
                      eval_metric=['logloss'],
                      early_stopping_rounds=20,
                      use_label_encoder=False,
                      subsample=0.5,
                      verbosity=0,
                      n_estimators=150
                      )
# fit model
model_rimAge.fit(x_train_rimAge, y_train)


# In[26]:


# "weight" is the number of times a feature appears in a tree.
feature_weights = model_rimAge.get_booster().get_score(importance_type='weight')
feature_weights = {k:v for k,v in sorted(feature_weights.items(), reverse=True, key=lambda item: item[1])}
group_data = list(feature_weights.values())
group_features = list(feature_weights.keys())
fig, ax = plt.subplots(figsize=(22,5), )
plt.style.use('classic')
plt.xticks(rotation=90)
plt.text(4, 500, "Number of times a feature appears in a tree", fontsize=24)
ax.bar(group_features, group_data)


# In[27]:


# "gain" is the average gain of splits which use the feature.
feature_gains = model_rimAge.get_booster().get_score(importance_type='gain')
feature_gains = {k:v for k,v in sorted(feature_gains.items(), reverse=True, key=lambda item: item[1])}
group_data = list(feature_gains.values())
group_features = list(feature_gains.keys())
fig, ax = plt.subplots(figsize=(22,5), )
plt.style.use('classic')
plt.xticks(rotation=90)
plt.text(4, 150, "Average gain of splits which use the feature", fontsize=24)
ax.bar(group_features, group_data)


# In[28]:


fig, ax = plt.subplots(figsize=(22,22))
plot_tree(model_rimAge, ax=ax)
plt.show()


# In[29]:


# Score all 'Active' accounts
# active_accounts.shape
active_accounts_RimAge = active_accounts.drop(['rim_age_reverse','cos', 'account_id', 'status'], axis=1)
scores = model_rimAge.predict_proba(active_accounts_RimAge)
scores_df = pd.DataFrame(scores, columns=['P_no_chargeOff', 'probability_score'])
scores_df['account_id'] = active_accounts['account_id'].reset_index(drop=True, inplace=False)
scores_df.drop('P_no_chargeOff', axis=1, inplace=True)


# In[42]:


scores_df2.head()


# In[43]:


#scores_df2.to_csv("mrm5_model_data2.csv", index=False)


# In[32]:


max(tmp['probability_score_n'])


# In[33]:


original_data = pd.read_csv("mrm5_model_data.csv")


# In[ ]:


mrm5_model_data_scored = pd.merge(original_data, scores_df2, on="account_id", how='inner')


# In[ ]:


mrm5_model_data_scored.head()


# In[ ]:


mrm5_model_data_scored.to_csv("mrm5_model_data_scored.csv", index=False)


# In[36]:


mrm5_model_data_scored.to_excel("mrm5_model_data_scored.xlsx", index=False)

