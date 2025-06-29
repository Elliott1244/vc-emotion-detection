import numpy as np
import pandas as pd
import pickle
import os
import yaml

params = yaml.safe_load(open('params.yaml', 'r'))['model_building']

from sklearn.ensemble import GradientBoostingClassifier

train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:,:-1].values
y_train = train_data.iloc[:,-1].values

clf = GradientBoostingClassifier(n_estimators = params['n_estimators'] , learning_rate = params['learning_rate'])

clf.fit(X_train,y_train)



os.makedirs("model", exist_ok=True)


with open("model/models.pkl", "wb") as f:
    pickle.dump(clf, f)







