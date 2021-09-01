# IMPORT PACKAGES
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# IMPORT DATA
churndf = pd.read_csv("D:\KaggleData\Data\churn_modeling.csv")

# LOOK AT THE TOP 5 AND BOTTOM 5 ROWS OF THE DATA SET
print(churndf.head())
print(churndf.tail())

# CHECK THE DATA TYPES
print(churndf.dtypes)

# TOTAL ROWS AND COLUMNS
print(churndf.shape)

# CHECK FOR DUPLICATE ROWS
duplicatedf = churndf[churndf.duplicated()]
print("Number of duplicate rows:", duplicatedf.shape)
# NO DUPLICATES

# CHECK FOR MISSING OR NULL VALUES
print(churndf.isnull().sum())
# NO NULLS

# REMOVE FIRST THREE COLUMNS OF THE DATASET BECAUSE IT ISNT RELEVANT
churndf = churndf.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# DETERMINE CURRENT PERCENTAGE OF THOSE WHO HAVE EXITED AND HAVE NOT EXITED
print(sum(churndf['Exited']) / len(churndf['Exited']))
print(1 - sum(churndf['Exited']) / len(churndf['Exited']))
# IF NOT EXITED IS PREDICTED EVERY TIME WE WOUILD BE RIGHT 79.63% OF THE TIME
# THIS IS THE TARGET THE MODEL IS TRYING TO BEAT

# TAKE OUT NON-NUMERICAL DATA
churn_num = churndf.drop(['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Exited'], axis=1)

# SET UP SNS STANDARD AESTHETICS FOR PLOTS
sns.set()

# MAKE BOXPLOTS FOR NUMERICAL DATA
for column in churn_num:
    sns.boxplot(x=churn_num[column])
    plt.show()

# SET BIN ARGUMENT
n_obs = len(churn_num)
n_bins = int(round(np.sqrt(n_obs), 0))
print(n_bins)

# CREATE HISTOGRAMS FOR EACH COLUMN
for column in churn_num:
    plt.hist(churn_num[column], density=True, bins=n_bins)
    plt.title(f"{column}")
    plt.show()

# CREATE A CORRELATION MATRIX
c = churn_num.corr()
print(c)

# FIND SUMMARY STATISTICS
print(churn_num.describe())