from sklearn.ensemble import RandomForestClassifier
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

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)

importances = rf.feature_importances_
plt.figure(figsize=(10, 10))
feature_importances = pd.Series(importances, index=X_train_transformed.columns)
feature_importances.nlargest(15).plot(kind='barh')
plt.show()

correlation_matrix = X_train_transformed.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

precision_rf, recall_rf, fscore_rf, _ = score(y_test, y_pred_rf, average='binary')
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

metrics_rf = pd.Series({'precision': precision_rf, 'recall': recall_rf, 
                        'fscore': fscore_rf, 'accuracy': accuracy_rf,
                        'auc': auc_rf})

print("\nRandom Forest Model Metrics:")
print(metrics_rf)

