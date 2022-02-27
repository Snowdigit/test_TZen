import pandas as pd
from training import scaler
from sauvegarde import model
from utils import processNa, encodage

#Chargement des données
datatest = pd.read_csv('../data/test_technical_test.csv')

#prepocessing
datatest= processNa(datatest)
datatest= encodage(datatest)

datatest = scaler.transform(datatest)

#prédiction
predictions = model.predict(datatest)
