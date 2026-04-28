import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("lung_cancer.csv")

# Drop non-feature columns
df = df.drop(columns=['index', 'Patient Id'], errors='ignore')

print("Shape:", df.shape)
print("\nClass distribution:")
print(df['Level'].value_counts())

# Separate features and labels
X = df.drop(columns=['Level'])
y = df['Level']

# Encode labels (High=0, Low=1, Medium=2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nLabel encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nTrain size:", X_train.shape[0])
print("Test size: ", X_test.shape[0])
