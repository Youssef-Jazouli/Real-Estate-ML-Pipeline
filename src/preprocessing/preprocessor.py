import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

def prepare_data(X_train, X_test, y_train, y_test, task_type="regression"):
    print(f"⏳ Normalisation et Encodage pour : {task_type}...")
    
    # Identifier les types de colonnes
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    # Pipelines de transformation
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    # Appliquer les transformations
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    
    # Équilibrer les classes avec SMOTE (uniquement pour la classification)
    if task_type == "classification":
        print("⚖️ Application de SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_clean, y_train = smote.fit_resample(X_train_clean, y_train)
        
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, f"models/preprocessor_{task_type}.pkl")
    
    return X_train_clean, X_test_clean, y_train, y_test