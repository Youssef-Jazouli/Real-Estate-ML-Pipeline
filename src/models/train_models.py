import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

def train_regression(X_train, y_train, X_test, y_test):
    print("📈 Entraînement du modèle de Régression...")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"📊 Régression -> MAE: {mae:.2f} | R²: {r2:.4f}")
    
    os.makedirs("models/regression", exist_ok=True)
    joblib.dump(model, "models/regression/best_regressor.pkl")

def train_classification(X_train, y_train, X_test, y_test):
    print("🧠 Entraînement du modèle de Classification...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"📊 Classification -> Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    
    os.makedirs("models/classification", exist_ok=True)
    joblib.dump(model, "models/classification/best_classifier.pkl")