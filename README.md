# Bank Customer Churn Prediction - Machine Learning Project

**[Test it here with your data](https://banck-customer-churn-prediction---machine-learning-project-awt.streamlit.app/)**

## 1. Project description

The goal of this project is to identify in a proactive manner the customers of a bank that might close their accounts. This goal is fueled by the fact that keeping existing clients is far cheaper than getting new ones, so the early identification of unhappy customers is critical.
This project will take the following steps: data analysis ,feature engineering, model comparison and selection and finaly, deploying the best model on an interactive web aplication.



## 2. Technologies and libraries used
* **Language:** Python
* **Pandas & NumPy:** Data Manipulation
* **Matplotlib & Seaborn:** Data Visualization
* **Scikit-Learn:** Preprocessing (StandardScaler, One-Hot Encoding), model training (Logistic Regression, Random Forest, MLPClassifier) and metrics(Confusion Matrix, Classification Report).
* **XGBoost:** fpr the gradient boosting algorithm.
* **Streamlit:** for the web aplication.
* **Joblib:** the saving and loading of the models (`.pkl`).

## 3. Exploratory Data Analysis (EDA) and Insights
* **Rate of churn:** The rate of customers that leave to customers to stay is about 1 to 4
* **Age:** The clients at the age between 45 and 55 have a higher rate of account closure than the younger ones.
* **Number of products:** The clients with 2 products bought tend to be more loyal than the ones with more products.
* **Balance:** There is a very high number of clients that have the balance 0 that tend to not close their accounts
* **Important features** the distributions for Tenure, Estimated Salary and Points Earned are flat, indicationg they're not crucial in predicting whether a client will close their account or not. The important features seem to be Balance, Age, Number of Products

## 4. Data Preprocessing

The following tendencies have been obeserved:

* **Data Leakage Prevention:** Am identificat și eliminat o capcană clasică în date - coloana `Complain` - care indica direct plecarea clientului și genera o acuratețe nerealistă de 100%. Am eliminat, de asemenea, coloanele fără valoare predictivă (`RowNumber`, `CustomerId`, `Surname`).
* **Categorical Encoding:** Am folosit `pd.get_dummies` (One-Hot Encoding) pentru a transforma variabilele text (`Geography`, `Gender`, `Card Type`) în format numeric, utilizând `drop_first=True` pentru a evita problema multicoliniarității.
* **Standardization:** Am scalat variabilele numerice folosind `StandardScaler` pentru a aduce toate caracteristicile la o scară comună (medie 0, deviație standard 1), pas esențial pentru antrenarea corectă a modelelor precum Regresia Logistică și Rețelele Neuronale.

## 5. Model training and testing
ultiple models have been trained, tested and compared. The #1 indicator was the recall score for the churned customers (because the dataset is imbalanced - so the accuracy can be misleading). The following results have been obtained:

1. **Logistic Regression:** Accuracy: 81%, Recall: 0.21. The model misses 79% of the churned clients
2. **Random Forest (Bagging):** Accuracy: 86.7%, Recall: 0.46. Still 64% of the true positives for churned class are missed.
3. **Multi-Layer Perceptron (Neural Network):** Accuracy: 80.6%, Recall: 0.52. The loss in accuracy indicates that there are more false positives.
4. **XGBoost (Gradient Boosting)** Accuracy: 83.20%, **Recall: 0.64**. The best trade-off for accuracy and recall

## 6. Determining the main causes of churn

We can analyse the most important features that lead to a client leaving the bank using the feature_importances_ attribute of the xgb model object.

By plotting those features from least important to most important we can take the following conlusions:

* The top 3 features that determine churn are: **The number of products** a client bought through the bank, weather the client is an **Active Member**, and the **Age** of the client.

* The bank seems to have a problem keeping clients in **Germany**.

* The type of card, the tenure or the salary don't seem to impact the clients decision tho leave the bank or not, but **Balance** or **Gender** appear to affect it (although not as much as the top 3).
