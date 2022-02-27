import joblib
from training import classifier

joblib.dump(classifier, '../models/xgb1.pkl')

model = joblib.load('../models/xgb1.pkl')