❤️ Heart Disease Prediction using Machine Learning
📌 Overview

This project focuses on building a machine learning model to predict the presence of heart disease using clinical and demographic data. The goal is to assist in early detection by analyzing patient health attributes.

RUN IN STREAMLIT:
cd path of the repo in a folder

Command:
Streamlit run app.py

📊 Dataset
Dataset contains 2181 patient records
Includes 14 features such as:
Age
Gender
Chest pain type (cp)
Cholesterol (chol)
Maximum heart rate (thalachh)
so on.....

🧠 Problem Statement

Predict whether a patient has heart disease (binary classification) based on medical attributes.

⚙️ Tech Stack
Python
Pandas, NumPy (data processing)
Scikit-learn (model building)
Matplotlib, Seaborn (visualization)

🔧 Data Preprocessing
Replaced missing values (?) with NaN
Applied mean imputation for numerical features
Applied most frequent imputation for categorical features
Converted categorical variables using one-hot encoding
Performed feature scaling after train-test split to avoid data leakage

🤖 Models Used
Logistic Regression
Support Vector Machine (SVM)
Decision Tree
Random Forest

📈 Model Performance
Model	Accuracy
Logistic Regression	~74%
SVM	~87%
Decision Tree	~92%
Random Forest	~93% ✅
Best Model: Random Forest
ROC-AUC Score: ~0.98

🛠️ Improvements
Reduced overfitting using hyperparameter tuning (max_depth, min_samples_split)
Used GridSearchCV for optimization
Evaluated models using Accuracy, F1-score, and ROC-AUC

🔍 Key Insights
Important features include:
thalachh (maximum heart rate)
oldpeak
age
cholesterol
Tree-based models performed better due to non-linear relationships in data

🚀 How to Run
# clone repo
git clone https://github.com/your-username/heart-disease-prediction.git

# go into folder
cd heart-disease-prediction

# install dependencies
pip install -r requirements.txt

# run script
python Heart_disease_Prediction.py

🧪 Sample Prediction
sample = X_test.iloc[0].to_dict()
predict(sample)

Output:

Heart Disease Detected / Normal

📌 Future Improvements
Add SHAP for model explainability
Deploy using Streamlit / Flask
Use advanced models like XGBoost
👩‍💻 Author

Likitha Sri Maddipatla
B.Tech CSE | Aspiring Data Scientist and ML Engineer
