# Importing required libraries.
import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from matplotlib.pyplot import figure
# %matplotlib inline
sns.set(color_codes=True)


data = pd.read_csv("mrm5_model_data.csv")

ct_cos_loan = pd.crosstab(data['cos'],data['loan'])
print(ct_cos_loan)

labels = ['COS NO/LOAN NO', 'COS NO/LOAN YES', 'COS YES/LOAN NO', 'COS YES/LOAN YES']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(ct_cos_loan, annot=labels, fmt='')


