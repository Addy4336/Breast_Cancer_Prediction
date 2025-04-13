import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             precision_score, recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, auc)
import pickle
import shap
import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv(r"C:\Users\lenovo\PycharmProjects\Breast_Cancer_Prediction\Dataset\data.csv")
df.drop(columns=["id", "Unnamed: 32"], inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
print("Missing Values:", df.isnull().sum().sum())

X = df.drop(columns=["diagnosis"])
Y = df["diagnosis"]


scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)
X_Scaled_Df = pd.DataFrame(X_Scaled, columns=X.columns)

print("Dataset shape after preprocessing:", X_Scaled_Df.shape)
print("Class distribution:\n", Y.value_counts())


corr_matrix = df.corr()
corr_with_target = corr_matrix["diagnosis"].abs().sort_values(ascending=False)
top_10_features = corr_with_target[1:11].index.tolist()
print("Top 10 features correlated with diagnosis:", corr_with_target[1:11])

X_Scaled_Df_Top10 = X_Scaled_Df[top_10_features]
X_train, X_test, Y_train, Y_test = train_test_split(X_Scaled_Df_Top10, Y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")



def evaluate_model(model, X_test, Y_test, model_name):
    Y_pred = model.predict(X_test)
    Y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    mcc = matthews_corrcoef(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    roc_auc = roc_auc_score(Y_test, Y_prob) if Y_prob is not None else "N/A"

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")
    if roc_auc != "N/A":
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(classification_report(Y_test, Y_pred))


    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{model_name}_cm.png")
    plt.close()


    if Y_prob is not None:
        fpr, tpr, _ = roc_curve(Y_test, Y_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f"{model_name}_roc.png")
        plt.close()



models = {
    "Logistic Regression": (LogisticRegression(), {"C": [0.001, 0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}),
    "Random Forest": (
    RandomForestClassifier(random_state=42), {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}),
    "SVM": (SVC(kernel="linear", probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "XGBoost": (XGBClassifier(eval_metric="logloss"),
                {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 6]})
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}

for name, (model, param_grid) in models.items():
    print(f"\nTuning {name}...")
    grid = GridSearchCV(model, param_grid, cv=skf, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, Y_train)
    best_models[name] = grid.best_estimator_

    print(f"Best Parameters for {name}:", grid.best_params_)
    print(f"Best Cross-Validation Accuracy: {grid.best_score_:.4f}")
    evaluate_model(grid.best_estimator_, X_test, Y_test, name)


best_xgb = best_models["XGBoost"]
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("shap_summary_bar.png")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png")
plt.close()

with open("best_breast_cancer_model.pkl", "wb") as file:
    pickle.dump(best_xgb, file)