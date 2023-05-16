import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle

# Read in the trades.csv file into a Pandas DataFrame
df = pd.read_csv('trades.csv')

# Check if all required columns are present in the DataFrame
required_cols = ['Price', 'Signal', 'Stop_Loss', 'Take_Profit', 'Open', 'High', 'Low', 'Close', 'Volume', 'Good_Breakout_Trade']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise ValueError(f"Required columns are missing in the DataFrame: {', '.join(missing_cols)}")

# Remove rows with missing values (NaN) from the DataFrame
df = df.dropna()

# Convert categorical features into one-hot encoding
if 'Signal' in df.columns:
    df = pd.get_dummies(df, columns=['Signal'])

# Prepare the data
features = [col for col in df.columns if col != 'Good_Breakout_Trade']
X = df[features]
y = df['Good_Breakout_Trade']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

hyperparameters = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000]
}


logistic_regression = LogisticRegression(random_state=42)


clf = GridSearchCV(logistic_regression, hyperparameters, cv=5, scoring='roc_auc')

try:
    clf.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"An error occurred during model training: {e}")
    print("Consider increasing the number of cross-validation folds or checking your code for potential errors.")

y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:,1])

print(f"Best hyperparameters: {clf.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC Score: {auc_roc}")

pickle.dump(clf, open('model.pkl', 'wb'))


