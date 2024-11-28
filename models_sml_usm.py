import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Load the Breast Cancer dataset (you can replace with other datasets as needed)
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0: malignant, 1: benign

# Feature and Target
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for algorithms like SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **1. Decision Tree Classifier (Classification)**

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
dt_pred = dt_model.predict(X_test_scaled)
print(f"\nDecision Tree Accuracy: {accuracy_score(y_test, dt_pred)}")  # Expected: Accuracy score


# **2. Random Forest Classifier (Classification)**

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
rf_pred = rf_model.predict(X_test_scaled)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, rf_pred)}")  # Expected: Accuracy score


# **3. Support Vector Machine (SVM) (Classification)**

# Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear', random_state=42)  # Use 'linear' kernel
svm_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
svm_pred = svm_model.predict(X_test_scaled)
print(f"\nSVM Accuracy: {accuracy_score(y_test, svm_pred)}")  # Expected: Accuracy score


# **4. K-Nearest Neighbors (KNN) (Classification)**

# K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust k
knn_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
knn_pred = knn_model.predict(X_test_scaled)
print(f"\nKNN Accuracy: {accuracy_score(y_test, knn_pred)}")  # Expected: Accuracy score


# **5. Naive Bayes (Classification)**

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
nb_pred = nb_model.predict(X_test_scaled)
print(f"\nNaive Bayes Accuracy: {accuracy_score(y_test, nb_pred)}")  # Expected: Accuracy score


# **6. Linear Regression (Regression - Predicting continuous values)**

# Linear Regression (for continuous outcomes, could be cancer progression score or survival time)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)  # Here y_train would be continuous in this case

# Predict and Evaluate
linear_pred = linear_model.predict(X_test_scaled)
print(f"\nLinear Regression R^2: {r2_score(y_test, linear_pred)}")  # Expected: R^2 score


# **7. Decision Tree Regressor (Regression)**

# Decision Tree Regressor (for continuous outcomes)
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train_scaled, y_train)  # Here y_train would be continuous

# Predict and Evaluate
dt_regressor_pred = dt_regressor.predict(X_test_scaled)
print(f"\nDecision Tree Regressor R^2: {r2_score(y_test, dt_regressor_pred)}")  # Expected: R^2 score


# **8. Random Forest Regressor (Regression)**

# Random Forest Regressor (for continuous outcomes)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Predict and Evaluate
rf_regressor_pred = rf_regressor.predict(X_test_scaled)
print(f"\nRandom Forest Regressor R^2: {r2_score(y_test, rf_regressor_pred)}")  # Expected: R^2 score


# **9. Support Vector Regression (SVR) (Regression)**

# Support Vector Regression (SVR)
svr_model = SVR(kernel='linear')
svr_model.fit(X_train_scaled, y_train)  # Again, y_train should be continuous for regression

# Predict and Evaluate
svr_pred = svr_model.predict(X_test_scaled)
print(f"\nSVR R^2: {r2_score(y_test, svr_pred)}")  # Expected: R^2 score


# **10. K-Means Clustering (Unsupervised - to cluster data)**

# K-Means Clustering (for grouping similar cancer types, for example)
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters for benign/malignant (for example)
kmeans.fit(X_train_scaled)  # Note: no target labels, just the features

# Print cluster centers
print(f"\nK-Means Cluster Centers: {kmeans.cluster_centers_}")

# Predict clusters
kmeans_pred = kmeans.predict(X_test_scaled)
print(f"Clusters predicted by KMeans: {np.unique(kmeans_pred)}")  # Expected: 2 clusters (benign and malignant)


# **Model Evaluation Summary:**
print("\nModel Evaluation Summary:")

# **Classification Models (Accuracy)**
print(f"Accuracy (Decision Tree): {accuracy_score(y_test, dt_pred)}")
print(f"Accuracy (Random Forest): {accuracy_score(y_test, rf_pred)}")
print(f"Accuracy (SVM): {accuracy_score(y_test, svm_pred)}")
print(f"Accuracy (KNN): {accuracy_score(y_test, knn_pred)}")
print(f"Accuracy (Naive Bayes): {accuracy_score(y_test, nb_pred)}")

# **Regression Models (R^2)**
print(f"R^2 (Linear Regression): {r2_score(y_test, linear_pred)}")
print(f"R^2 (Decision Tree Regressor): {r2_score(y_test, dt_regressor_pred)}")
print(f"R^2 (Random Forest Regressor): {r2_score(y_test, rf_regressor_pred)}")
print(f"R^2 (SVR): {r2_score(y_test, svr_pred)}")
