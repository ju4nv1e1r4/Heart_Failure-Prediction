# %%
import pickle
import datetime

import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score


# %%
df = pd.read_csv('../../data/heart_cleaned.csv', delimiter=';')
df.head(3)
# %%
df2 = pd.DataFrame.copy(df)

df2['Sex'].replace({'M': 0, 'F': 1}, inplace=True)
df2['ChestPainType'].replace({'TA':0, 'ATA':1, 'NAP':2, 'ASY': 3}, inplace=True)
df2['RestingECG'].replace({'Normal':0, 'ST':1, 'LVH':2}, inplace=True)
df2['ExerciseAngina'].replace({'N':0, 'Y':1}, inplace=True)
df2['ST_Slope'].replace({'Up':0, 'Flat':1, 'Down':2}, inplace=True)

# %%
df2.head()

# %%
pred = df2.iloc[:, 0:11].values
pred

# %%
pred.shape

# %%
target = df2.iloc[:, 11].values
target

# %%
target.shape

# %%
pred_esc = StandardScaler().fit_transform(pred)
pred_esc

# %%
pred_df = pd.DataFrame(pred_esc)
pred_df

# %%
pred_df.describe()

# %%
pred2 = df.iloc[:, 0:11].values
pred2

# %%
pred2[:,1] = LabelEncoder().fit_transform(pred[:,1])
pred2

# %%
pred2[:,2] = LabelEncoder().fit_transform(pred[:,2])
pred2[:,6] = LabelEncoder().fit_transform(pred[:,6])
pred2[:,8] = LabelEncoder().fit_transform(pred[:,8])
pred2[:,10] = LabelEncoder().fit_transform(pred[:,10])

pred2

# %%
pred3 = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),
                                         [1,2,6,8,10])],
                                         remainder='passthrough').fit_transform(pred2)

# %%
pred3.shape

# %%
pred3esc = StandardScaler().fit_transform(pred3)
pred3esc

# %%
pca = PCA(n_components=4)
predictors_pca = pca.fit_transform(pred2)

predictors_pca.shape

# %%
pca.explained_variance_ratio_

# %%
kpca = KernelPCA(n_components=4, kernel='rbf')
predictors_kernel = kpca.fit_transform(pred2)

predictors_kernel.shape

# %%
lda = LinearDiscriminantAnalysis(n_components=1)
predictors_lda = lda.fit_transform(pred2, target)

predictors_lda.shape

# %%
lda.explained_variance_ratio_

# %%
arq = open('heart.pkl', 'wb')
pickle.dump(target, arq)
arq2 = open('heart2.pkl', 'wb')
pickle.dump(pred, arq2)
arq3 = open('heart3.pkl', 'wb')
pickle.dump(pred_esc, arq3)
arq4 = open('heart4.pkl', 'wb')
pickle.dump(pred2, arq4)
arq5 = open('heart5.pkl', 'wb')
pickle.dump(pred3, arq5)
arq6 = open('heart6.pkl', 'wb')
pickle.dump(pred3esc, arq6)

# %%
X_train, X_test, y_train, y_test = train_test_split(pred3esc,
                                                    target, 
                                                    test_size=0.3,
                                                    random_state=0)

X_train.shape

# %%
X_test.shape

# %%
y_test.shape

# %%
y_train.shape

# %%
naive = GaussianNB()
naive.fit(X_train, y_train)

# %%
pred_naive = naive.predict(X_test)
pred_naive

# %%
accuracy_score(y_test, pred_naive)

# %%
confusion_matrix(y_test, pred_naive)

# %%
print(classification_report(y_test, pred_naive))

# %%
pred_train = naive.predict(X_train)
pred_train

# %%
accuracy_score(y_train, pred_train)

# %%
confusion_matrix(y_train, pred_train)

# %%
kfold = KFold(n_splits = 30, shuffle=True, random_state=5)

# %%
model = GaussianNB()
result = cross_val_score(model, pred3esc, target, cv = kfold)

result.mean()

# %%
svm = SVC(kernel='rbf', random_state=1, C=2)
svm.fit(X_train, y_train)

# %%
pred_svm = svm.predict(X_test)
pred_svm

# %%
accuracy_score(y_test, pred_svm)

# %%
confusion_matrix(y_test, pred_svm)

# %%
print(classification_report(y_test, pred_svm))

# %%
pred_train_svm = svm.predict(X_train)
pred_train_svm

# %%
accuracy_score(y_train, pred_train_svm)

# %%
confusion_matrix(y_train, pred_train_svm)

# %%
kfold_svm = KFold(n_splits = 30, shuffle=True, random_state=5)

# %%
model_svm = SVC(kernel='rbf', random_state=1, C=2)
result_svm = cross_val_score(model_svm, pred3esc, target, cv = kfold_svm)

result_svm.mean()

# %%
logreg = LogisticRegression(random_state=1,
                            max_iter=600,
                            penalty='l2',
                            tol=0.0001,C=1,
                            solver='lbfgs')
logreg.fit(X_train, y_train)

# %%
pred_logreg = logreg.predict(X_test)
pred_logreg

# %%
accuracy_score(y_test, pred_logreg)

# %%
confusion_matrix(y_test, pred_logreg)

# %%
print(classification_report(y_test, pred_logreg))

# %%
pred_train_logreg = logreg.predict(X_train)
pred_train_logreg

# %%
accuracy_score(y_train, pred_train_logreg)

# %%
confusion_matrix(y_train, pred_train_logreg)

# %%
kfold_logreg = KFold(n_splits = 30, shuffle=True, random_state=5)

# %%
model_logreg = LogisticRegression(random_state=1,
                                max_iter=600,
                                penalty='l2',
                                tol=0.0001,C=1,
                                solver='lbfgs')
result_logreg = cross_val_score(model_logreg, pred3esc, target, cv = kfold_logreg)

result_logreg.mean()

# %%
