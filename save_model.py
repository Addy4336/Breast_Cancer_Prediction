import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\Downloads\data.csv")  # Ensure the correct dataset path

# Drop unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Encode target variable (M = 1, B = 0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Separate features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train the best Logistic Regression model
best_lr = LogisticRegression(C=0.1, solver="liblinear")
best_lr.fit(X_train, y_train)

# Save the trained model
with open("breast_cancer_model.pkl", "wb") as model_file:
    pickle.dump(best_lr, model_file)

print("Model saved successfully as 'breast_cancer_model.pkl'!")
