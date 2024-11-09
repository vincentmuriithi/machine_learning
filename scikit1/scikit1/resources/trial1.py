from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import  numpy as np

def run() -> None:
    model = LogisticRegression(solver="liblinear")
    cancer_data = load_breast_cancer()


    df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
    df['target'] = cancer_data['target']

    X = df[cancer_data.feature_names].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.score(X_test, y_test) * 100,"%", sep="")
    print("f1_score:",f1_score(y_test, y_pred))
