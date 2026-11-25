import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / 'german_credit.csv')
X = df[['Age', 'Credit Amount']]
y = df['Creditability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=42),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    results.append({'Model': name, 'Accuracy': accuracy_score(y_test, preds), 'F1': f1_score(y_test, preds), 'ROC-AUC': roc_auc_score(y_test, probs)})
results = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print(results.to_string(index=False))
winning_name = results.iloc[0]['Model']
joblib.dump(models[winning_name], BASE / 'creditability_model.joblib')
print(f'\nSaved: {winning_name} -> creditability_model.joblib')
results.to_csv(BASE / 'model_comparison.csv', index=False)
