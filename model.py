## Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import pickle

import warnings
warnings.filterwarnings('ignore')

## Import model libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


## Read in dataset
df = pd.read_csv('dataset.csv')

## Data glimpse
df.head()

## data info
df.info()

df.describe()

## Correlation heatmap
#corrmat = df.corr()
#top_corr_features = corrmat.index

#plt.figure(figsize=(12,8))

#g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')

## Select X andY features
df['target'].value_counts()

y = df['target']
X = df.drop(['target','oldpeak'],axis=1)

## Model development
np.random.seed(1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))

## Save Model
pickle.dump(rfc, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))