import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('heart-kaggle-refine.xlsx')
X = data.drop('target', axis=1)
y = data['target']

# Data Overview
print("\n" + "=" * 50)
print("Dataset overview")
print("=" * 50)
print(f"Number of patient records: {data.shape[0]}")
print(f"Number of clinical features: {data.shape[1] - 1}")
print(f"Missing values detected: {int(data.isnull().sum().sum())}")
print(
    f"Class distribution -> heart disease: {y.value_counts()[1]} "
    f"({y.value_counts()[1] / len(y) * 100:.1f}%), "
    f"no heart disease: {y.value_counts()[0]} "
    f"({y.value_counts()[0] / len(y) * 100:.1f}%)."
    )

numeric_df = data.select_dtypes(include=['number'])
for col in numeric_df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(numeric_df[col].dropna(), bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(f'histogram_{col}.png')
    plt.close()

# Pearson correlation matrix
correlation_matrix = data.corr(method='pearson')
plt.figure(figsize=(12, 10))
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation matrix")
plt.savefig('pearson_correlation_heatmap.png')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, y_train = X_scaled[:250], y[:250]
X_test, y_test = X_scaled[250:], y[250:]

# Using first 250 samples for training and the rest for testing
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(num_leaves=31, n_estimators=50, verbose=-1)
}
# Test-set performance of baseline models
print("\n" + "=" * 50)
print("Test-set performance of baseline models")
print("=" * 50)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name}: test accuracy = {accuracy:.1%}")
    print(classification_report(y_test, y_pred))

# K-Cross validation
print("\n" + "=" * 50)
print("5-fold cross-validation")
print("=" * 50)
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores.mean()
    print(f"\n{name}:")
    print(f"  Fold accuracies: {np.round(cv_scores, 3)}")
    print(f"  Mean accuracy: {cv_scores.mean():.1%}, standard deviation: {cv_scores.std():.1%}")

# Hyperparameter tuning
print("\n" + "=" * 50)
print("Hyperparameter tuning with Grid Search")
print("=" * 50)
print("\nDecision Tree")
param_grid_dt = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_dt.fit(X_train, y_train)
print(f"  Best parameters: {grid_dt.best_params_}")
print(f"  Best cross-validation accuracy: {grid_dt.best_score_:.1%}")
print("\nRandom Forest")
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)
print(f"  Best parameters: {grid_rf.best_params_}")
print(f"  Best cross-validation accuracy: {grid_rf.best_score_:.1%}")
print("\nLogistic Regression")
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)
print(f"  Best parameters: {grid_lr.best_params_}")
print(f"  Best cross-validation accuracy: {grid_lr.best_score_:.1%}")

# Confusion matrices
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.show()
plt.show()

# ROC curves
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} AUC = {roc_auc}')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves for all models')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.show()

# Model accuracy comparison
model_names = list(models.keys())
accuracies = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies)
plt.ylabel('Accuracy')
plt.title('Test accuracy by model')
plt.xticks(rotation=45)
plt.savefig('model_accuracy_comparison.png')
plt.show()

# Cross-validation score comparison
plt.figure(figsize=(10, 6))
plt.bar(list(cv_results.keys()), list(cv_results.values()))
plt.ylabel('Mean CV accuracy')
plt.title('Cross-validation scores')
plt.xticks(rotation=45)
plt.savefig('cv_scores_comparison.png')
plt.show()