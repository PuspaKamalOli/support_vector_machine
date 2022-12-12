import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

cancer = load_breast_cancer()
print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df_feat.head()

X = df_feat
y = cancer['target']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
svm = SVC()
svm.fit(x_train, y_train)

pred = svm.predict(x_test)

print(confusion_matrix(y_test, pred))
print('/n')
print(classification_report(y_test, pred))

# here we notice everything is predicted  on class 1 so we use grid use for better performance

param_grid = {'C': [0.1, 1, 10, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, ]}
grid = GridSearchCV(SVC(), param_grid, verbose=True)
grid.fit(x_train, y_train)
predictions = grid.predict(x_test)
print(grid.best_params_)
print(grid.best_estimator_)

confusion_matrix(y_test, predictions)

print(classification_report(y_test, predictions))
