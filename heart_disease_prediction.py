import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_excel('heart-kaggle-refine.xlsx')

X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train = X_scaled[:250]
y_train = y[:250]
X_test = X_scaled[250:]
y_test = y[250:]


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    # 'LightGBM': LGBMClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)                    
    y_pred = model.predict(X_test)                 
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred))
