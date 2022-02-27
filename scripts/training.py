import numpy as np
import pandas as pd


from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from  sklearn.preprocessing import LabelEncoder

from utils import processNa, encodage, renameTarget

#importation
data = pd.read_csv('../data/train_technical_test.csv')

#prepocessing
data= processNa(data)
data= encodage(data)
data= renameTarget(data)

X = data.drop('Target', axis=1).values
y = data['Target']

scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Utlisation de xgboost
classifier = XGBClassifier(max_depth=6, objective='multi:softmax', n_estimators=1000, verbosity=0)
classifier.fit(X, y)


