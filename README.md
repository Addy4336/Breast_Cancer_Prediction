<<<<<<< HEAD
Breast Cancer Classification Using Machine Learning
Overview
This project focuses on classifying breast cancer cases as benign or malignant using the Wisconsin Breast Cancer Dataset (WBCD) from the UCI Machine Learning Repository. The dataset contains 569 samples with 30 features, and the project employs machine learning models to achieve high classification accuracy, with a particular emphasis on recall to minimize false negatives in a medical context. The project includes data preprocessing, feature selection, model training, hyperparameter tuning, and a Streamlit web application for interactive predictions.

Features
Dataset: Wisconsin Breast Cancer Dataset (569 samples, 30 features, no missing values)
Preprocessing: Feature selection based on correlation, standardization of features
Models Trained:
Logistic Regression (Accuracy: 98.25%, Recall: 97.67%)
Support Vector Machine (SVM) (Accuracy: 97.37%, Recall: 95.35%)
Random Forest (Accuracy: 95.61%, Recall: 93.02%)
XGBoost (Accuracy: 94.74%, Recall: 90.70%)
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Specificity, Matthews Correlation Coefficient (MCC), ROC-AUC
Visualization: ROC curves, confusion matrices, and feature importance plots
Web Application: Streamlit app for real-time predictions

Project Structure:

breast_cancer_classification/
│
├── data/
│   └── breast_cancer_data.csv  # Raw dataset (downloaded from UCI)
├── notebooks/
│   └── analysis.ipynb           # Jupyter notebook for data exploration and model training
├── src/
│   ├── preprocess.py           # Data preprocessing and feature selection
│   ├── train_models.py         # Model training and hyperparameter tuning
│   ├── evaluate.py             # Model evaluation and visualizations
│   └── app.py                  # Streamlit web application
├── models/
│   ├── logistic_regression.pkl # Trained Logistic Regression model
│   ├── svm.pkl                 # Trained SVM model
│   ├── random_forest.pkl       # Trained Random Forest model
│   └── xgboost.pkl             # Trained XGBoost model
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file

Prerequisites:
Python 3.8+
Git
Virtual environment (recommended)
Installation
Clone the Repository (once hosted on GitHub):

git clone https://github.com/your-username/breast_cancer_classification.git
cd breast_cancer_classification
Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:


pip install -r requirements.txt
The requirements.txt should include:

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
streamlit
shap

Download the Dataset:

The dataset is available at UCI Machine Learning Repository.
Place breast_cancer_data.csv in the data/ folder or update the data loading path in preprocess.py.
Usage
Run the Analysis Notebook:
Open notebooks/analysis.ipynb in Jupyter Notebook or JupyterLab to explore the data, train models, and visualize results.
Execute cells sequentially to preprocess data, train models, and generate evaluation metrics and plots.

streamlit run src/app.py
Access the app at http://localhost:8501 in your browser.
Input feature values to get real-time predictions from the trained Logistic Regression model.
Results
Dataset Shape: 569 samples, 30 features (reduced to top 10 based on correlation)
Class Distribution:
Benign (0): 357
Malignant (1): 212
Top Features:
concave points_worst (correlation: 0.793566)
perimeter_worst (0.782914)
concave points_mean (0.776614)
...
Model Performance:
Logistic Regression: Best parameters {'C': 10, 'solver': 'liblinear'}, Accuracy: 98.25%, ROC-AUC: 0.9984
SVM: Best parameters {'C': 10, 'kernel': 'rbf'}, Accuracy: 97.37%, ROC-AUC: 0.9954
Random Forest: Best parameters {'max_depth': 5, 'n_estimators': 100}, Accuracy: 95.61%, ROC-AUC: 0.9954
XGBoost: Best parameters {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 50}, Accuracy: 94.74%, ROC-AUC: 0.9925
Key Insight: Logistic Regression outperforms others, with high recall (97.67%) critical for minimizing false negatives in cancer diagnosis.

Future Enhancements
Implement k-fold cross-validation for robust model evaluation
Address class imbalance using SMOTE or weighted loss functions
Add model interpretability with SHAP or feature importance analysis
Expand to include imaging data for a hybrid ML-DL approach
Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the  file for details.

Acknowledgments
Dataset: UCI Machine Learning Repository
Libraries: scikit-learn, XGBoost, Streamlit, pandas, matplotlib, seaborn
Inspiration: Research on ML for medical diagnosis
Contact
For questions, reach out to adityamishra7990@gmail.com or open an issue on GitHub.
=======
# 🩺 Breast Cancer Prediction App 🚀  
This project is a **Machine Learning-powered Web App** that predicts whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** based on clinical data.  

🔗 **Live App:** [Click here to access the app](https://your-streamlit-app-link.com) *(Update after deployment)*  

---

## 📊 **Project Overview**
Breast cancer is a leading cause of mortality in women. Early detection through **Machine Learning** can help in timely treatment. This app allows users to input tumor characteristics and get an instant prediction.  

✅ **Trained on the Breast Cancer Wisconsin Dataset (569 samples, 30 features).**  
✅ **Best model: Logistic Regression (Tuned) with 99.12% accuracy.**  
✅ **Web app built using `Streamlit` and deployed on Streamlit Cloud.**  

---

## 🛠 **Tech Stack**
- **Programming Language:** Python 🐍  
- **Machine Learning:** `Scikit-Learn`, `Logistic Regression`  
- **Web Framework:** `Streamlit`  
- **Deployment:** `Streamlit Cloud`, `GitHub`  

---

## 📌 **How to Run the Project Locally?**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/Breast_Cancer_Prediction.git
cd Breast_Cancer_Prediction
>>>>>>> bda0e1936681376fafd9e9672eda759217d5f032
