from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

app = Flask(__name__)

# ── Train model on startup ──────────────────────────────────────────
def train_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df_model = df[features + ['Survived']].copy()
    df_model['Age'].fillna(df_model['Age'].median(), inplace=True)
    df_model['Embarked'].fillna(df_model['Embarked'].mode()[0], inplace=True)
    df_model['Sex'] = df_model['Sex'].map({'male': 0, 'female': 1})
    df_model['Embarked'] = df_model['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    X = df_model[features]
    y = df_model['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Survival stats for EDA charts
    survival_by_sex    = df.groupby('Sex')['Survived'].mean().round(2).to_dict()
    survival_by_class  = df.groupby('Pclass')['Survived'].mean().round(2).to_dict()
    feature_importance = dict(zip(features, model.feature_importances_.round(4).tolist()))

    return model, acc, cm, survival_by_sex, survival_by_class, feature_importance

print("Training model…")
MODEL, ACCURACY, CM, SURV_SEX, SURV_CLASS, FEAT_IMP = train_model()
print(f"Model ready — accuracy: {ACCURACY}%")

# ── Routes ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           accuracy=ACCURACY,
                           cm=CM,
                           surv_sex=SURV_SEX,
                           surv_class=SURV_CLASS,
                           feat_imp=FEAT_IMP)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    passenger = pd.DataFrame([{
        'Pclass':   int(data['pclass']),
        'Sex':      int(data['sex']),
        'Age':      float(data['age']),
        'SibSp':    int(data['sibsp']),
        'Parch':    int(data['parch']),
        'Fare':     float(data['fare']),
        'Embarked': int(data['embarked'])
    }])
    pred = MODEL.predict(passenger)[0]
    prob = MODEL.predict_proba(passenger)[0][int(pred)]
    return jsonify({'survived': int(pred), 'probability': round(float(prob) * 100, 1)})

if __name__ == '__main__':
    app.run(debug=True)
