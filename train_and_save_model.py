import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def main():
    print("Loading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'lung_cancer.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    X = df.drop(columns=['index', 'Patient Id', 'Level'], errors='ignore')
    le = LabelEncoder()
    y = le.fit_transform(df['Level'])  # High=0, Low=1, Medium=2
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target classes: {le.classes_}")
    
    # Train-test split (matching original model scripts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train XGBoost model
    print("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Verify performance on test data
    X_test_scaled = scaler.transform(X_test)
    accuracy = xgb_model.score(X_test_scaled, y_test)
    print(f"Model test accuracy: {accuracy * 100:.2f}%")
    
    # Save the model, scaler, and classes list
    model_dir = base_dir
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    classes_path = os.path.join(model_dir, 'classes.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(classes_path, 'wb') as f:
        pickle.dump(le.classes_, f)
        
    print(f"Successfully saved model to {model_path}")
    print(f"Successfully saved scaler to {scaler_path}")
    print(f"Successfully saved classes list to {classes_path}")

if __name__ == '__main__':
    main()
