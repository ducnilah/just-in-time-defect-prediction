import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                             accuracy_score, confusion_matrix, precision_recall_fscore_support as score)

def load_datasets(data):
    train = pd.read_csv(os.path.join(data, 'train.csv'))
    test = pd.read_csv(os.path.join(data, 'test.csv'))
    val = pd.read_csv(os.path.join(data, 'val.csv'))
    return train, test, val

def apply_transformations(df):
    transformed_df = df.copy()
    for feature in ['date','ns','nf','ndev','nuc','sexp','la', 'ld', 'lt', 'age', 'exp', 'rexp']:
        if feature in transformed_df.columns:
            transformed_df[feature] = transformed_df[feature] + 1 
            transformed_df[feature], _ = boxcox(transformed_df[feature])
    
    return transformed_df

data = 'features'
df_train, df_test, df_val = load_datasets(data)
for df in [df_train, df_test, df_val]:
    df.drop(columns=['Unnamed: 0', '_id'], inplace=True, errors='ignore')

X_train = df_train.drop(columns=['bug'])
y_train = df_train['bug']
X_test = df_test.drop(columns=['bug'])
y_test = df_test['bug']
X_val = df_val.drop(columns=['bug'])
y_val = df_val['bug']

X_train_transformed = apply_transformations(X_train)
X_test_transformed = apply_transformations(X_test)
X_val_transformed = apply_transformations(X_val)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_transformed)
X_test_scaled = scaler.transform(X_test_transformed)
X_val_scaled = scaler.transform(X_val_transformed)

param_grid = [{
    'penalty' : ['l2','l1','elsaticnet', 'none'],
    'C': np.logspace(-4,4,20),
    'solver': ['saga'],
    'max_iter' : [1000, 2000, 5000]
}]
lr = LogisticRegression(solver='saga', max_iter=3000, class_weight='balanced')

grid_search = GridSearchCV(lr, param_grid, cv=4, scoring='roc_auc', verbose=True, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_lr = grid_search.best_estimator_

y_pred = best_lr.predict(X_test_scaled)
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]

precision, recall, fscore, _ = score(y_test, y_pred, average='binary')
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

metrics = pd.Series({'precision': precision, 'recall': recall, 
                     'fscore': fscore, 'accuracy': accuracy,
                     'auc': auc})

print("Best Model Hyperparameters:")
print(grid_search.best_params_)

print("\nModel Metrics:")
print(metrics)