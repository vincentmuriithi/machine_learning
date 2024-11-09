from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import  numpy as np

model = LogisticRegression(solver="liblinear")
cancer_data = load_breast_cancer()


df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values

model.fit(X, y)
y_pred = model.predict(X)
print(confusion_matrix(y, y_pred))
