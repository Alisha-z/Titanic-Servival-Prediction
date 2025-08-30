# Titanic-Servival-Prediction

# 🚢 Titanic Survival Prediction (Classification)

## 📌 Objective
The aim of this project is to **predict the survival status of passengers aboard the Titanic** using machine learning models. By analyzing demographic and ticket-related data, we classify whether a passenger survived (`1`) or not (`0`).

## 📂 Dataset
- **Kaggle Titanic Dataset**  
  Includes passenger details such as:
  - PassengerId  
  - Name  
  - Sex  
  - Age  
  - Ticket Class (Pclass)  
  - Fare  
  - Embarked  
  - Survived (Target Variable)

## ⚙️ Technologies & Libraries
- **Python 3.x**
- **Pandas** – data preprocessing & cleaning  
- **NumPy** – numerical computations  
- **Matplotlib / Seaborn** – data visualization & EDA  
- **Scikit-learn** – machine learning models & evaluation  
  - Logistic Regression  
  - Decision Tree Classifier  
  - Metrics: Accuracy, Precision, Recall, Confusion Matrix, ROC Curve  

## 🧠 Methods & Approach
1. **Data Preprocessing**
   - Handled missing values (Age, Cabin, Embarked)  
   - Encoded categorical features using `LabelEncoder` and `pd.get_dummies()`  
   - Normalized numerical features where necessary  

2. **Exploratory Data Analysis (EDA)**
   - Visualized survival distribution across gender, class, and age  
   - Correlation analysis between features  

3. **Model Training**
   - Trained **Logistic Regression** as a baseline  
   - Built **Decision Tree Classifier** for interpretability  
   - Split dataset into **train/test sets**  

4. **Model Evaluation**
   - Evaluated models using:
     - Accuracy  
     - Precision & Recall  
     - Confusion Matrix  
     - ROC Curve  

## 🎯 Learning Outcomes
- Handling missing and categorical data in real-world datasets  
- Understanding supervised binary classification problems  
- Evaluating model performance using multiple metrics  
- Building interpretable ML models for prediction tasks  


