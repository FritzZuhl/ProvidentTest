#!/usr/bin/env python
# coding: utf-8

# ### Provident Credit Union Exercise  
# > #### Analyst: Frederick $($Fritz$)$ Zuhl
# > #### December 17, 2021
# #### Part   1 - EDA and Feature Preparation
# + #### Check data from completness and general sanity check
# + #### EDA and Univarite Analysis - Correlate with Target Variable (Charge-Off)
# + #### Apply Binning when needed to increase statistical significance and over-fitting
# *** 
# ***
# <br>
# <br>

# In[1]:


# Import libraries.
import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# ***
# ***
# ### Part1 / Section 1 - Read in the data
# + #### General Sanity Checks
# + #### Is data complete? Check for outliers.
# ***

# In[2]:


data = pd.read_csv("mrm5_model_data.csv")


# In[3]:


print(data.info)


# In[4]:


data.describe()


# In[5]:


# What are the fields in the data
data.columns.tolist()


# In[6]:


data.head()


# In[7]:


data.tail()


# ***
# ***
# ### Part 1 / Section 2
# + #### Looking at the Variables Individually.
# + #### How do they compare/correlate/relate to target value COS?
# + #### Feature Engineering - Transform some fields to improve their predictability and prevent over-fitting within a classification models.
# 
# ##### Note: For speed, I used Excel as a kind of scratch pad as a side tool to look at these fields, and for some common univariate analysis. The Excel file is not included, but is available upon request.
# ***

# In[8]:


# Helper Functions
# These functions are used throughout this EDA, and bring a general 'feel' of the data.
# They check for completeness, range, outliers, other errors.

import matplotlib.pyplot as plt

def plot_catVar(var_name):
    valueCounts = data[var_name].value_counts()
    ind = list(valueCounts.index)
    val = list(valueCounts.values)
    fig = plt.figure(figsize=(30, 10))
    plt.bar(ind, val, color='blue', width=0.4)
    plt.ylabel("Counts")
    plt.xlabel(var_name.upper())
    plt.title("Counts of {} values".format(var_name.upper()))
    plt.show()

def check_catVar(var_name):
    unique_values = data[var_name].unique()
    valueCounts = data[var_name].value_counts()
    print("*** For the field {} ***".format(var_name))
    print("\nUnique values:\n", unique_values)
    print("\nThe value counts are:")
    print(valueCounts)
    print("\nCount of NULL {} values:".format(var_name), data[var_name].isnull().sum())
    print("Count of NA {} values:".format(var_name), data[var_name].isna().sum())
    print("Plot out Values")
    plot_catVar(var_name)

def check_contVar(var_name):
    print("\nCount of NULL {} values:".format(var_name), data[var_name].isnull().sum())
    print("Count of NA {} values:".format(var_name), data[var_name].isna().sum())
    print("Max value of {}:".format(var_name), data[var_name].max())
    print("Min value of {}:".format(var_name), data[var_name].min())
    print("Type of value", type(data[var_name].values))
    print("Random sample of values:\n", data[var_name].sample(n=10))

def CrossTabCheck(field):
    print(data[field].value_counts())
    print("\n")

    print("Normalize by bin, or portion of charge-offs.".upper())
    ct1 = pd.crosstab(data['cos'], data[field]).apply(lambda r: r / r.sum(), axis=0)
    print(ct1)

    print("\nNormalize by row, or portion of members.".upper())
    ct2 = pd.crosstab(data['cos'], data[field]).apply(lambda r: r / r.sum(), axis=1)
    print(ct2)

    print("\nDirect Counts.".upper())
    ct3 = pd.crosstab(data['cos'], data[field])
    print(ct3)


# ***
# ***
# ### <span style="color:red">TARGET: COS </span>
# #### Account Charge-Off Indicator
# #### Dependent or Target Variable
# ***

# In[9]:


check_catVar('cos')


# ***
# ***
# ### <span style="color:red">ACCOUNT_ID </span>
# ***

# In[10]:


# Independent Field: account_id
# Is account_id unique and complete?
dups = data['account_id'].duplicated()
print("Number of duplicate account_id: ", dups.sum())
# Finding NULL values
print("Number of missing account_id:", data["account_id"].isnull().sum())


# Observation: The 'account-id' field is complete and unique. It can be used as an index for data.

# * **
# ***
# ### <span style="color:red">1: STATUS </span>
# #### Latest status of account.
# ***

# In[11]:


check_catVar('status')


# In[12]:


# How does 'status' correlates with 'cos'?


# In[13]:


pd.crosstab(data['cos'],data['status'])


# Discussion on STATUS field:
# All cos positive accounts are closed. Not all closed account are positive 'cos'.
# 516/7235 = 0.071, or %7.1 are closed and cos positive.

# * **
# ***
# ### <span style="color:red">2: LOAN</span>
# #### Does account have any loan product associated with it?
# ***

# In[14]:


check_catVar('loan')


# In[15]:


CrossTabCheck('loan')


# ***
# ### Calculations done on excel spreadsheet (not shown here)
# #### P(cos yes | loan=1)	    0.00110
# #### P(cos yes | loan=0)	    0.01236 (no loan associated with account)
# 
# ### Feature Transform? None.

# * **
# ***
# ### <span style="color:red">3: S_PLUS_C</span>
# #### Is type of account include only savings, or both savings and checking?
# ***

# In[16]:


check_catVar('s_plus_c')


# In[17]:


# Cross Tab with target field.
CrossTabCheck('s_plus_c')


# ***
# EDA for 's_plus_c' variable was done outside of this notebook (not shown here)  
# Prob(cos yes  |  s_plus_c = 1)	=  0.011356717  
# Prob(cos yes  |  s_plus_c = 0)	=  0.004603175  
# If an account has a checking account, it is much more likely to result in charge off than saving alone.  
# Feature Transform: None

# * **
# ***
# ### <span style="color:red">4: NTRIGGERS</span>
# #### Number of alerts from other financial institutions.
# ***

# In[18]:


check_catVar('ntriggers')


# In[19]:


# Cross Tab with Target Varaible
ct_cos_ntriggers = pd.crosstab(data['cos'],data['ntriggers'])
print(ct_cos_ntriggers)


# In[20]:


def bin_ntriggers(x):
    if x==0:
        return "nt_zero"
    if 0<x<2:
        return "nt_small"
    else:
        return "nt_large"

data['ntriggers_bin'] = data['ntriggers'].apply(bin_ntriggers)

CrossTabCheck('ntriggers_bin')


# ***
# Feature Transform: Bin into nt_zero, nt_small, nt_large

# ***
# ***
# ### <span style="color:red">5: NUM_TIMES_NEG</span>
# #### Number of time the account went negative.
# ***

# In[21]:


check_catVar('num_times_neg')


# In[22]:


# Cross Tab with Target Variable
ct_cos_numtimeneg = pd.crosstab(data['cos'],data['num_times_neg'])
print(ct_cos_numtimeneg)


# In[23]:


# Bin values to make them statistically significant.
data['num_times_neg_bin'] = pd.cut(data['num_times_neg'], bins=(0, 2, 25, 26, 49, 300),
                                  labels=['bin1', 'bin2', 'bin3', 'bin4', 'bin5'], right=False, include_lowest=True,
                                  ordered=False)


# In[24]:


CrossTabCheck('num_times_neg_bin')


# * **
# ***
# ### <span style="color:red">6: MAX_DAYS_NEG</span>
# #### Maximum consecutive days account was negative.
# ***

# In[25]:


check_catVar('max_days_neg')


# In[26]:


# Using Excel, inspect how max_days_neg field correlates with charged-off accounts.
# Modeling tactic: bin the values of max_days_neg to create statistical significance.

def bin_max_days_neg(x):
    if 0 <= x < 2:
        return 'bin1'
    if 2 <= x < 5:
        return 'bin2'
    if 5 <= x < 30:
        return 'bin3'
    if 30 <= x < 110:
        return 'bin4'
    else:
        return 'bin5'

data['max_days_neg_bin'] = data['max_days_neg'].apply(bin_max_days_neg)

CrossTabCheck('max_days_neg_bin')


# Note on the transformation:
# Use function bin_max_days_neg() to increase statistical significance and prevent over-fitting.

# * **
# ***
# ### <span style="color:red">7: RIM_AGE</span>
# #### Age of the accout owner's membership in months
# ***

# In[27]:


check_contVar('rim_age')


# In[28]:


plt.hist(data['rim_age'], bins=42)
plt.rcParams['figure.figsize'] = (30,10)
plt.show()


# In[29]:


# what is the distribution of rim_age among those that have COS?
data_cosPOS = data[data['cos'].values == 1].copy(deep=True)
plt.hist(data_cosPOS['rim_age'], bins=42)
plt.rcParams['figure.figsize'] = (30,10)
plt.show()


# In[30]:


max_month = max(data['rim_age'])

def reverse_rim_age(x):
    return max_month - x

data['rim_age_reverse'] = data['rim_age'].apply(reverse_rim_age)

data_cosPOS = data[data['cos'].values == 1].copy(deep=True)
plt.hist(data_cosPOS['rim_age_reverse'], bins=42)
plt.rcParams['figure.figsize'] = (30,10)
plt.show()


# The distribution of rim_age with COS is counter to what we would expect. I wonder if the age value has been reversed. Test when the rim_age values are reversed.
# Possible transform: f(rim_age) = max(rim_age) - rim_age

# * **
# ***
# ### <span style="color:red">8: CK_RETURNS</span>
# #### Number of returned checks since Jan. 2017
# ***

# In[31]:


check_contVar('ck_returns')


# In[32]:


data['ck_returns'] = data['ck_returns'].fillna(0)


# In[33]:


check_contVar("ck_returns")


# In[34]:


plt.hist(data['ck_returns'], bins=42)
plt.rcParams['figure.figsize'] = (30,10)
plt.show()


# In[35]:


tmp = data[data['ck_returns'] != 0]
tmp['ck_returns']


# In[36]:


def bin_ckreturns(x):
    if x==0:
        return 'bin1'
    if 1<=x<2:
        return 'bin2'
    if 2<=x<7:
        return 'bin3'
    if x>=7:
        return 'bin4'

data['ck_returns_bin'] = data['ck_returns'].apply(bin_ckreturns)
CrossTabCheck('ck_returns_bin')


# * **
# ***
# ### <span style="color:red">9: FICO_B</span>
# #### FICO credit score
# ***

# In[37]:


check_contVar("fico_b")


# In[38]:


plt.hist(data['fico_b'], bins=100)
plt.rcParams['figure.figsize'] = (30,10)
plt.show()


# In[39]:


# what is the distribution of rim_age among those that have COS?
data_cosPOS = data[data['cos'].values == 1].copy(deep=True)
plt.hist(data_cosPOS['fico_b'], bins=100)
plt.rcParams['figure.figsize'] = (30,10)
plt.show()


# In[40]:


# How many fico scores are 0?
COSall_FICO_Zero = data[data['fico_b'] == 0]
print("Count of FICO scores = 0: ", COSall_FICO_Zero.shape[0])
COSpos_FICO_Zero = data[(data['fico_b'] == 0) & (data['cos'] == 1)]
print("Count of FICO scores = 0 AND COS positive: ", COSpos_FICO_Zero.shape[0])


# In[41]:


def fico_bin_f(x):
    bin = 'bin0'
    if x==0:
        bin = 'bin1'
    if 0 < x < 480:
        bin = 'bin2'
    if 480 <= x < 550:
        bin = 'bin3'
    if 550 <= x < 600:
        bin = 'bin4'
    if 600 <= x < 680:
        bin = 'bin5'
    if 680 <= x < 740:
        bin = 'bin6'
    if 740 <= x < 860:
        bin = 'bin7'
    return bin

data['fico_bin'] = data['fico_b'].apply(fico_bin_f)

CrossTabCheck('fico_bin')


# #### Note: Fico_b needs to be binned to make the results more statistical significant.

# * **
# ***
# ### <span style="color:red">10 and 11: AVG_BAL6 and AVG_BAL3</span>
# #### Rolling 6-month/3-month deposit amount
# ***

# In[42]:


check_contVar('avg_bal6')


# In[43]:


# Consider NA or missing values as 0.
data['avg_bal6'] = data['avg_bal6'].fillna(0)
data['avg_bal3'] = data['avg_bal3'].fillna(0)
check_contVar('avg_bal6')
check_contVar('avg_bal3')


# In[44]:


plt.hist(data['avg_bal6'], bins=300)
plt.rcParams['figure.figsize'] = (30,10)
plt.show()


# In[45]:


# Binning avg_bal6, avg_bal3
def bin(x):
    if x<0:
        return "negative"
    if x==0:
        return 'zero'
    if x>0:
        return 'positive'

data['avg_bal6_binned'] = data['avg_bal6'].apply(bin)
data['avg_bal3_binned'] = data['avg_bal3'].apply(bin)


# In[46]:


field = 'avg_bal6_binned'

print(data[field].value_counts())
print("\n")

print("Normalize by bin, or portion of charge-offs.".upper())
ct1 = pd.crosstab(data['cos'], data[field]).apply(lambda r:r/r.sum(), axis=0)
print(ct1)

print("\nNormalize by row, or portion of members.".upper())
ct2 = pd.crosstab(data['cos'], data[field]).apply(lambda r:r/r.sum(), axis=1)
print(ct2)

print("\nDirect Counts.".upper())
ct3 = pd.crosstab(data['cos'], data[field])
print(ct3)


# In[47]:


CrossTabCheck("avg_bal3_binned")


# *** 
# ***
# ### End of Part 1
# #### Review data
# #### Write out to CSV file for Part 2 - Modeling
# ***

# In[1]:


# Review data preparation up to this point.
print(data.columns)


# In[49]:


# Output
data.to_csv("data_with_features.csv", index=False)

