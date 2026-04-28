import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Load & prepare
df = pd.read_csv("lung_cancer.csv")
df = df.drop(columns=['index', 'Patient Id'], errors='ignore')
X = df.drop(columns=['Level'])
le = LabelEncoder()
y = le.fit_transform(df['Level'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN":                  KNeighborsClassifier(n_neighbors=7),
    "SVM":                  SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost":              xgb.XGBClassifier(random_state=42, eval_metric='mlogloss',
                                              use_label_encoder=False)
}

# Train and collect results
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall":    recall_score(y_test, y_pred, average='weighted'),
        "F1 Score":  f1_score(y_test, y_pred, average='weighted')
    })

# Results table
final_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(final_df.to_string(index=False))

# Grouped bar chart
df_melted = final_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

palette = {
    "Logistic Regression": "#90cdf4",
    "KNN":                  "#b794f4",
    "SVM":                  "#fbd38d",
    "XGBoost":              "#68d391"
}

ax = sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, palette=palette)

plt.title("Performance Comparison: Lung Cancer Risk Prediction",
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
plt.xlabel("Evaluation Metrics", fontsize=12)
plt.ylim(0.7, 1.05)
plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9), textcoords='offset points',
                fontsize=9, fontweight='bold')

sns.despine()
plt.tight_layout()
plt.show()
