# 🚢 Titanic Survival Predictor

A binary classification ML project that predicts whether a Titanic passenger would have survived, based on features like age, gender, ticket class and fare. Includes an interactive Flask web app for live predictions.

## Features
- Live survival predictor with confidence percentage
- Confusion matrix on the 20% test set
- EDA charts: survival by gender and by class
- Feature importance visualization

## Tech Stack
- **Language:** Python
- **ML:** scikit-learn (Random Forest Classifier), pandas, NumPy
- **Web:** Flask, HTML/CSS, Chart.js

## Model
- **Algorithm:** Random Forest (100 decision trees, `random_state=42`)
- **Features:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Target:** Survived (0 = No, 1 = Yes)
- **Split:** 80% train / 20% test
- **Accuracy:** XX%

## Preprocessing
- Missing `Age` filled with the median
- Missing `Embarked` filled with the mode
- `Sex` and `Embarked` encoded as numbers

## Key Findings
- Females had a much higher survival rate than males
- 1st class passengers survived more than 3rd class
- Sex and Fare were among the most important features

## How to Run
```bash
git clone https://github.com/NotZaid18/Titanic-ML-project.git
cd Titanic-ML-project
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Future Work
- Compare with Logistic Regression, SVM and XGBoost
- Hyperparameter tuning with GridSearchCV
- Feature engineering (passenger titles, family size)
- Deploy publicly on Render or Railway

## Author
Syed Zaid, [@NotZaid18](https://github.com/NotZaid18)
