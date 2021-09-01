# IMPORT PACKAGES
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
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

# CONVERT STRINGS TO NUMBERS
geography_mapping = {
    'France': 0, 
    'Spain': 1,
    'Germany': 2,
}
churndf.Geography = [geography_mapping[item] for item in churndf.Geography]

gender_mapping = {
    'Female': 0, 
    'Male': 1,
}
churndf.Gender = [gender_mapping[item] for item in churndf.Gender]

# SPLIT DATA INTO TRAIN AND TEST SETS
y = churndf['Exited']
X = churndf.drop('Exited', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# FIT THE MODEL
churn_mod = sm.Logit(y_train, X_train)
churn_mod2 = churn_mod.fit()

# FIND MODEL RESULTS
print(churn_mod2.summary())

# CREATE A FUNCTION TO DETERMINE OPTIMAL CUTOFF
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

# GET TRAINING PREDICTIONS
train_pred = churn_mod2.predict(X_train)

# FIND OPTIMAL CUT OFF
cut_off = Find_Optimal_Cutoff(y_train, train_pred)
cut_off = float(cut_off[0])

# OPTIMAL CUT OFF [0.20970714248577862]
train_pred1 = []

for pred in train_pred:
    if pred > cut_off:
        train_pred1.append(1)
    else:
        train_pred1.append(0)

# PRINT CONFUSION MATRIX
print(confusion_matrix(y_train, train_pred1))

# DETERMINE WHETHER OR NOT THE PREDICTION WAS CORRECT AND FIGURE OUT ACCURACY
correct_predictions = y_train == train_pred1
print(sum(correct_predictions) / len(correct_predictions))

# FIND ACCURACY FOR TEST PREDICTIONS
test_pred = churn_mod2.predict(X_test)

test_pred1 = []

for pred in test_pred:
    if pred > cut_off:
        test_pred1.append(1)
    else:
        test_pred1.append(0)

print(confusion_matrix(y_test, test_pred1))

correct_predictions = y_test == test_pred1
print(sum(correct_predictions) / len(correct_predictions))
