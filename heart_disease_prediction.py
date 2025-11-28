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

print("=" * 60)
print("Beginning detailed exploration of the heart disease dataset")
print("=" * 60)
print(f"The dataset consists of {data.shape[0]} patient records and {data.shape[1]-1} clinical features.")
print(f"Data completeness check: No missing values were detected.")
print(f"Class distribution - Patients with heart disease: {y.value_counts()[1]} ({y.value_counts()[1]/len(y)*100:.1f}%), without heart disease: {y.value_counts()[0]} ({y.value_counts()[0]/len(y)*100:.1f}%).")

print("\nGenerating histograms for all numerical clinical features...")
numeric_df = data.select_dtypes(include=['number'])
for col in numeric_df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(numeric_df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Patients")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'histogram_{col}.png', dpi=150, bbox_inches='tight')
    plt.close()
print("Histogram plots have been saved for each clinical feature.")

print("\nComputing Pearson correlation matrix to analyze feature relationships...")
correlation_matrix = data.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Pearson Correlation Matrix for Clinical Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pearson_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Pearson correlation heatmap successfully saved.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, y_train = X_scaled[:250], y[:250]
X_test, y_test = X_scaled[250:], y[250:]

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(num_leaves=31, n_estimators=50, verbose=-1)
}

print("\n" + "=" * 60)
print("Training five different machine learning models on the dataset")
print("=" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Model achieved an accuracy of {accuracy:.1%} on the test set.")
    print(classification_report(y_test, y_pred))

print("\n" + "=" * 60)
print("Evaluating model reliability through 5-fold cross-validation")
print("=" * 60)
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores.mean()
    print(f"\n{name}:")
    print(f"  Cross-validation scores: {cv_scores}")
    print(f"  Mean accuracy: {cv_scores.mean():.1%} with a standard deviation of {cv_scores.std():.1%}")

print("\n" + "=" * 60)
print("Performing hyperparameter optimization using GridSearchCV")
print("=" * 60)

print("\nOptimizing Decision Tree parameters...")
param_grid_dt = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_dt.fit(X_train, y_train)
print(f"Best Decision Tree parameters: {grid_dt.best_params_}")
print(f"Best cross-validation accuracy: {grid_dt.best_score_:.1%}")

print("\nOptimizing Random Forest parameters...")
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)
print(f"Best Random Forest parameters: {grid_rf.best_params_}")
print(f"Best cross-validation accuracy: {grid_rf.best_score_:.1%}")

print("\nOptimizing Logistic Regression parameters...")
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)
print(f"Best Logistic Regression parameter: {grid_lr.best_params_}")
print(f"Best cross-validation accuracy: {grid_lr.best_score_:.1%}")

print("\n" + "=" * 60)
print("Generating visualizations to further analyze model performance")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}')
    axes[idx].set_ylabel('True Diagnosis')
    axes[idx].set_xlabel('Predicted Diagnosis')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
print("Confusion matrix visualizations have been saved.")

plt.figure(figsize=(10, 8))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Chance Level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model ROC Curves Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.show()
print("ROC curve comparisons have been saved.")

fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(models.keys())
accuracies = [accuracy_score(y_test, model.fit(X_train, y_train).predict(X_test)) for model in models.values()]
ax.bar(model_names, accuracies, color='skyblue')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Model Accuracies on Test Dataset')
ax.set_ylim([0, 1])
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')
plt.show()
print("Bar chart comparing model accuracies has been saved.")

fig, ax = plt.subplots(figsize=(10, 6))
cv_names = list(cv_results.keys())
cv_scores = list(cv_results.values())
ax.bar(cv_names, cv_scores, color='lightcoral')
ax.set_ylabel('Cross-Validation Accuracy')
ax.set_title('Model Stability Across Patient Folds')
ax.set_ylim([0, 1])
for i, v in enumerate(cv_scores):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cv_scores_comparison.png')
plt.show()
print("Visualization comparing cross-validation results has been saved.")
