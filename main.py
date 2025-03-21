import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv (r"C:\Users\lenovo\Downloads\data.csv")
# df.info()
# print (df.head())

# Drop unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)
# df.info()

df["diagnosis"] = df["diagnosis"].map({"M":1,"B": 0})

print ("Missing Values:", df.isnull().sum().sum())

X= df.drop(columns = ["diagnosis"])
Y = df["diagnosis"]

# Standardize features
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

# Convert into Dataframe
X_Scaled_Df = pd.DataFrame(X_Scaled,columns= X.columns)


print("Dataset shape after preprocessing:", X_Scaled_Df.shape)

# # Plot class distribution
# plt.figure(figsize=(6, 4))
# sns.countplot(X=Y, palette=["green", "red"])
# plt.title("Class Distribution: Benign vs Malignant", fontsize=14)
# plt.xlabel("Diagnosis (0 = Benign, 1 = Malignant)", fontsize=12)
# plt.ylabel("Count", fontsize=12)
#plt.show()

# Print value counts
print("Class distribution:\n", Y.value_counts())

corr_matrix = df.corr()

# Plot heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
# plt.title("Feature Correlation Heatmap", fontsize=14)
# plt.show()

# Get correlation with the target variable
corr_with_target = corr_matrix["diagnosis"].abs().sort_values(ascending=False)
print("Top 10 features correlated with diagnosis:", corr_with_target[1:11])

X_train,X_test, Y_train,Y_test = train_test_split(X_Scaled_Df,Y, test_size = 0.2, random_state= 42)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Applying logistic Regression
LR_Model = LogisticRegression()
LR_Model.fit (X_train, Y_train)

# Prediction using the model
Y_Pred_LR = LR_Model.predict(X_test)


# Model evaluation
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(Y_test, Y_Pred_LR))
print("ROC-AUC Score:", roc_auc_score(Y_test, Y_Pred_LR))
print(classification_report(Y_test, Y_Pred_LR))

# Applying Random Forest Classifier
Rf_Model = RandomForestClassifier(n_estimators= 100, random_state=42)
Rf_Model.fit(X_train,Y_train)

# Predictions
Y_Pred_RF =Rf_Model.predict(X_test)
# Model evaluation
print("Random Forest Results:")
print("Accuracy:", accuracy_score(Y_test, Y_Pred_RF))
print("ROC-AUC Score:", roc_auc_score(Y_test, Y_Pred_RF))
print(classification_report(Y_test, Y_Pred_RF))

# Using Support Vector Machine

SVC_Model = SVC(kernel="linear", probability=True)
SVC_Model.fit(X_train, Y_train)

Y_Pred_SVC =SVC_Model.predict(X_test)

# Model evaluation
print("SVM Results:")
print("Accuracy:", accuracy_score(Y_test, Y_Pred_SVC))
print("ROC-AUC Score:", roc_auc_score(Y_test, Y_Pred_SVC))
print(classification_report(Y_test, Y_Pred_SVC))

# Using XGBoost CLassifier
XGB_Model = XGBClassifier()
XGB_Model.fit (X_train, Y_train)

# Prediction
Y_Pred_XGB =XGB_Model.predict(X_test)
# Model evaluation
print("XGBoost Results:")
print("Accuracy:", accuracy_score(Y_test,Y_Pred_XGB))
print("ROC-AUC Score:", roc_auc_score(Y_test, Y_Pred_XGB))
print(classification_report(Y_test, Y_Pred_XGB))

# Store results in a dictionary
model_results = {
    "Logistic Regression": accuracy_score(Y_test, Y_Pred_LR),
    "Random Forest": accuracy_score(Y_test, Y_Pred_RF),
    "SVM": accuracy_score(Y_test, Y_Pred_SVC),
    "XGBoost": accuracy_score(Y_test, Y_Pred_XGB),
}

# Convert to DataFrame
results_df = pd.DataFrame(model_results.items(), columns=["Model", "Accuracy"])
print(results_df.sort_values(by="Accuracy", ascending=False))

Param_Grid_LR = {
    "C": [0.001, 0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

Grid_LR = GridSearchCV(LR_Model, Param_Grid_LR, cv=5,scoring="accuracy",n_jobs=-1)
Grid_LR.fit(X_train, Y_train)
print ("Best parameter for Logistic Regression:",Grid_LR.best_params_)
print ("Best Accuracy:", Grid_LR.best_score_)

Best_LR = Grid_LR.best_estimator_
Y_Pred_LR_Tuned = Best_LR.predict(X_test)

print("Tuned Logistic Regression Results:")
print("Accuracy:", accuracy_score(Y_test, Y_Pred_LR_Tuned))
print("ROC-AUC Score:", roc_auc_score(Y_test, Y_Pred_LR_Tuned))
print(classification_report(Y_test, Y_Pred_LR_Tuned))



# Define hyperparameters for Random Forest Algorithm
param_Grid_RF = {
    "n_estimators": [50, 100, 200],  # Number of trees
    "max_depth": [5, 10, 20, None],  # Depth of trees
    "min_samples_split": [2, 5, 10],  # Minimum samples needed to split a node
    "min_samples_leaf": [1, 2, 4],  # Minimum samples in a leaf
    "bootstrap": [True, False]  # Use bootstrap sampling
}

# Hyper parameter tuning for random forest classifier

# Grid Search
Grid_RF = GridSearchCV(Rf_Model, param_Grid_RF, cv=5, scoring="accuracy", n_jobs=-1)
Grid_RF.fit(X_train, Y_train)

# Best parameters & accuracy
print("Best Parameters for Random Forest:", Grid_RF.best_params_)
print("Best Accuracy:", Grid_RF.best_score_)

# Train the best model
best_RF = Grid_RF.best_estimator_
Y_Pred_RF_tuned = best_RF.predict(X_test)
# Evaluate performance
print("Tuned Random Forest Results:")
print("Accuracy:", accuracy_score(Y_test, Y_Pred_RF_tuned))
print("ROC-AUC Score:", roc_auc_score(Y_test, Y_Pred_RF_tuned))
print(classification_report(Y_test, Y_Pred_RF_tuned))


# Define hyperparameter grid
param_grid_svm = {
    "C": [0.1, 1, 10],  # Regularization strength
    "kernel": ["linear", "rbf"],  # Type of kernel
    "gamma": ["scale", "auto"]  # Kernel coefficient
}

# Create model


# Grid Search
grid_svm = GridSearchCV(SVC_Model, param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1)
grid_svm.fit(X_train, Y_train)

# Best parameters & accuracy
print("Best Parameters for SVM:", grid_svm.best_params_)
print("Best Accuracy:", grid_svm.best_score_)

# Train the best model
best_svm = grid_svm.best_estimator_
y_pred_svm_tuned = best_svm.predict(X_test)

# Evaluate performance
print("Tuned SVM Results:")
print("Accuracy:", accuracy_score(Y_test, y_pred_svm_tuned))
print("ROC-AUC Score:", roc_auc_score(Y_test, y_pred_svm_tuned))
print(classification_report(Y_test, y_pred_svm_tuned))



# Define hyperparameter grid for XGB
param_grid_xgb = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 6, 10],
    "subsample": [0.7, 1.0]
}

# Create model
xgb = XGBClassifier(eval_metric="logloss")

# Grid Search
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring="accuracy", n_jobs=-1)
grid_xgb.fit(X_train, Y_train)

# Best parameters & accuracy
print("Best Parameters for XGBoost:", grid_xgb.best_params_)
print("Best Accuracy:", grid_xgb.best_score_)

# Train the best model
best_xgb = grid_xgb.best_estimator_
y_pred_xgb_tuned = best_xgb.predict(X_test)

# Evaluate performance
print("Tuned XGBoost Results:")
print("Accuracy:", accuracy_score(Y_test, y_pred_xgb_tuned))
print("ROC-AUC Score:", roc_auc_score(Y_test, y_pred_xgb_tuned))
print(classification_report(Y_test, y_pred_xgb_tuned))



