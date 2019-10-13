import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Checking files in directory
import os
for dirname, _, filenames in os.walk('C:/Users/gurdsin/Desktop/ml/python/logistic reg/weather dataset and code'):
for filename in filenames:
print(os.path.join(dirname, filename))

#importing data
data = 'C:/Users/gurdsin/Desktop/ml/python/logistic reg/weather dataset and code/weatherAUS.csv'
dataset1 = pd.read_csv(data)

#view data
dataset1.info()
#Dataset consist of numerical(type: object) and categorical(type: float64) variables

#properties of dataset
dataset1.describe()
dataset1.describe(include=['object']) #check for only categorical variables

#univariate analysis
dataset1['RainTomorrow'].isnull().sum()
dataset1['RainTomorrow'].nunique()
dataset1['RainTomorrow'].unique()
dataset1['RainTomorrow'].value_counts()
dataset1['RainTomorrow'].value_counts()/len(dataset1)

f, ax = plt.subplots(figsize=(6, 8))
ax = sns.countplot(x="RainTomorrow", data=dataset1, palette="Set1")
plt.show()

#We need to remove RISK_MM because we want to predict 'RainTomorrow' and RISK_MM can leak some info to our model
dataset1 = dataset1.drop(columns=['RISK_MM'],axis=1)

#bivariate analysis
categorical = [var for var in dataset1.columns if dataset1[var].dtype=='O']
print(categorical)

#check for null values
dataset1[categorical].isnull().sum()

# view frequency of categorical variables
for var in categorical:
print(dataset1[var].value_counts())

# view frequency distribution of categorical variables
for var in categorical:
print(dataset1[var].value_counts()/np.float(len(dataset1)))

# check for cardinality in categorical variables (high cardinality not good for ML)

for var in categorical:
print(var, ' contains ', len(dataset1[var].unique()), ' labels')

#convert 'Date' to date format
dataset1['Date'] = pd.to_datetime(dataset1['Date'])

#extracting date/month/year
dataset1['Year'] = dataset1['Date'].dt.year
dataset1['Month'] = dataset1['Date'].dt.month
dataset1['Day'] = dataset1['Date'].dt.day
dataset1.drop('Date', axis=1, inplace = True)

# get k-1 dummy variables after One Hot Encoding (categorical variables)
# also add an additional dummy variable to indicate there was missing data
pd.get_dummies(dataset1.Location, drop_first=True).head()
pd.get_dummies(dataset1.WindGustDir, drop_first=True, dummy_na=True).head()
pd.get_dummies(dataset1.WindDir9am, drop_first=True, dummy_na=True).head()

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category
pd.get_dummies(dataset1.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
d.get_dummies(dataset1.WindDir3pm, drop_first=True, dummy_na=True).head()
pd.get_dummies(dataset1.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
pd.get_dummies(dataset1.RainToday, drop_first=True, dummy_na=True).head()
pd.get_dummies(dataset1.RainToday, drop_first=True, dummy_na=True).sum(axis=0)

#checking for numerical variables
numerical = [var for var in dataset1.columns if dataset1[var].dtype!='O']
print(numerical)

#checking for outliers
print(round(dataset1[numerical].describe()),2)

#looks like Rainfall, Evaporation, WindSpeed9am and WindSpeed3pm columns may contain outliers
# draw boxplots to visualize outliers
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = dataset1.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2, 2, 2)
fig = dataset1.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2, 2, 3)
fig = dataset1.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2, 2, 4)
fig = dataset1.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')

#check for distribution
#If the variable follows normal distribution,
#then I will do Extreme Value Analysis otherwise if they are skewed, I will find IQR (Interquantile range)
# plot histogram to check distribution

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = dataset1.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 2)
fig = dataset1.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 3)
fig = dataset1.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 4)
fig = dataset1.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')

# Since all the four variables are skewed. So, I will use interquantile range to find outliers.
# find outliers for Rainfall variable
IQR = dataset1.Rainfall.quantile(0.75) - dataset1.Rainfall.quantile(0.25)
Lower_fence = dataset1.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = dataset1.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for Evaporation variable
IQR = dataset1.Evaporation.quantile(0.75) - dataset1.Evaporation.quantile(0.25)
Lower_fence = dataset1.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = dataset1.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for WindSpeed9am variable
IQR = dataset1.WindSpeed9am.quantile(0.75) - dataset1.WindSpeed9am.quantile(0.25)
Lower_fence = dataset1.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = dataset1.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for WindSpeed3pm variable
IQR = dataset1.WindSpeed3pm.quantile(0.75) - dataset1.WindSpeed3pm.quantile(0.25)
Lower_fence = dataset1.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = dataset1.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
#outliers can also be removed by using below meathod:
#its time to remove the outliers in our data - we are using Z-score to detect and remove the outliers.

#Multivariate Analysis
#discover relationship between variables using heat map and pair plots
correlation = dataset1.corr()
plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.show()

 
#MinTemp and MaxTemp variables are highly positively correlated (correlation coefficient = 0.74).
#MinTemp and Temp3pm variables are also highly positively correlated (correlation coefficient = 0.71).
#MinTemp and Temp9am variables are strongly positively correlated (correlation coefficient = 0.90).
#MaxTemp and Temp9am variables are strongly positively correlated (correlation coefficient = 0.89).
#MaxTemp and Temp3pm variables are also strongly positively correlated (correlation coefficient = 0.98).
#WindGustSpeed and WindSpeed3pm variables are highly positively correlated (correlation coefficient = 0.69).
#Pressure9am and Pressure3pm variables are strongly positively correlated (correlation coefficient = 0.96).
#Temp9am and Temp3pm variables are strongly positively correlated (correlation coefficient = 0.86).

#pair plots
num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']
sns.pairplot(dataset1[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')
plt.show()

#declaring feature vector and target variable
X = dataset1.drop(['RainTomorrow'], axis=1)
y = dataset1['RainTomorrow']

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Engineering is the process of transforming raw data into useful features that help us to
#understand our model better and increase its predictive power

# check missing values in numerical variables in X_train
X_train[numerical].isnull().sum()
# check missing values in numerical variables in X_test
X_test[numerical].isnull().sum()

# print percentage of missing values in the numerical variables in training set
for col in numerical:
if X_train[col].isnull().mean()>0:
print(col, round(X_train[col].isnull().mean(),4))

#Assumptions: Data are missing completely at random (MCAR).
#There are two methods which can be used to impute missing values.
#One is mean or median imputation and other one is random sample imputation.
#When there are outliers in the dataset, we should use median imputation.
#So, I will use median imputation because median imputation is robust to outliers.

#I will impute missing values with the appropriate statistical measures of the data, in this case median.
#Imputation should be done over the training set, and then propagated to the test set.
#It means that the statistical measures to be used to fill missing values both in train and test set,
#should be extracted from the train set only. This is to avoid overfitting.

# impute missing values in X_train and X_test with respective column median in X_train
for df1 in [X_train, X_test]:
for col in numerical:
col_median=X_train[col].median()
df1[col].fillna(col_median, inplace=True)

# check again missing values in numerical variables
X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()

# print percentage of missing values in the categorical variables in training set
X_train[categorical].isnull().mean()
# print categorical variables with missing data
for col in categorical:
if X_train[col].isnull().mean()>0:
print(col, (X_train[col].isnull().mean()))

# impute missing categorical variables with most frequent value
for df2 in [X_train, X_test]:
df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

# check missing values in categorical variables in X_train
X_train[categorical].isnull().sum()
X_test[categorical].isnull().sum()

#Removing outliers from numerical variabvles
#We'll use top coding approach to cap maximum values and remove outliers above those
def max_value(df3, variable, top):
return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

X_train.Rainfall.max(), X_test.Rainfall.max()

#Lets deal with the categorical cloumns now
# simply change yes/no to 1/0 for RainToday and RainTomorrow
dataset1['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
dataset1['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

#Feature scaling
cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

#Model Training
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)
# fit the model
logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)
y_pred_test

#we can select the important variable using the below meathod
#now that we are done with the pre-processing part, let's see which are the important features for RainTomorrow!
#Using SelectKBest to get the top features!
from sklearn.feature_selection import SelectKBest, chi2
X = dataset1.loc[:,dataset1.columns!='RainTomorrow']
y = dataset1[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)]) #top 3 columns

#Let's get hold of the important features as assign them as X
#df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
#X = df[['Humidity3pm']] # let's use only one feature Humidity3pm
#y = df[['RainTomorrow']]

#check accuracy score
from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

#compare the train-test and test-set accuracy
y_pred_train = logreg.predict(X_train)
y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# fit the Logsitic Regression model with C=100
# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)

# fit the model
logreg100.fit(X_train, y_train)

#check for overfitting and underfitting
# print the scores on training and test set
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))

#Two scores are very close, so no overfitting.
#Let's increase the value of C and see what's the result
# fit the Logsitic Regression model with C=100
# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)

# fit the model
logreg100.fit(X_train, y_train)

# print the scores on training and test set
print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))

#We can see that, C=100 results in higher test set accuracy and also a
#slightly increased training set accuracy. So, we can conclude that a more
#complex model should perform better.

#Now, I will investigate, what happens if we use more regularized model than
#the default value of C=1, by setting C=0.01.

# fit the Logsitic Regression model with C=001
# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)

# fit the model
logreg001.fit(X_train, y_train)

# print the scores on training and test set
print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))

#So, if we use more regularized model by setting C=0.01, then both the training
#and test set accuracy decrease relative to the default parameters.

#Compare model accuracy with null accuracy
#So, the model accuracy is 0.8501. But, we cannot say that our model is very
#good based on the above accuracy. We must compare it with the null accuracy.
#Null accuracy is the accuracy that could be achieved by always predicting the
#most frequent class.

#So, we should first check the class distribution in the test set.
# check class distribution in test set
y_test.value_counts()

# check null accuracy score
null_accuracy = (22067/(22067+6372))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

#Interpretation
#We can see that our model accuracy score is 0.8501 but null accuracy score
#is 0.7759. So, we can conclude that our Logistic Regression model is doing
#a very good job in predicting the class labels.

#Confusion Matrix
# Print the Confusion Matrix and slice it into four pieces
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#Classification report is another way to evaluate the classification model
#performance. It displays the precision, recall, f1 and support scores
#for the model.

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# print classification accuracy
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# print classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
#recall score
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
#true positive rate
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
#false positive rate
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
#specificity
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))