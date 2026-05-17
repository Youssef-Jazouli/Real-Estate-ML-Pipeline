import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
from src.extraction.db_connector import extract_obt
from src.features.engineering import split_and_engineer
from src.preprocessing.preprocessor import prepare_data
from src.models.train_models import train_regression, train_classification

def run_pipeline():
    print("🚀 DÉMARRAGE DU PIPELINE ML IMMOBILIER 🚀\n")
    
    # 1. Extraction
    if not os.path.exists("data/raw/obt_data.csv"):
        df_raw = extract_obt()
    else:
        print("🔄 Données brutes trouvées localement.")
        df_raw = pd.read_csv("data/raw/obt_data.csv")
        
    # 2. Split & Feature Engineering
    train_df, test_df = split_and_engineer(df_raw)
    
    # ---------------------------------------------------------
    # PIPELINE RÉGRESSION (Prédire prix_mad ou log_prix)
    # ---------------------------------------------------------
    print("\n" + "-"*40 + "\n1️⃣ PIPELINE RÉGRESSION\n" + "-"*40)
    target_reg = 'log_prix'
    cols_to_drop_reg = ['prix_mad', 'price_category', target_reg]
    
    X_train_reg = train_df.drop(columns=cols_to_drop_reg, errors='ignore')
    X_test_reg = test_df.drop(columns=cols_to_drop_reg, errors='ignore')
    
    X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = prepare_data(
        X_train_reg, X_test_reg, train_df[target_reg], test_df[target_reg], "regression"
    )
    train_regression(X_tr_reg, y_tr_reg, X_te_reg, y_te_reg)
    
    # ---------------------------------------------------------
    # PIPELINE CLASSIFICATION (Prédire price_category)
    # ---------------------------------------------------------
    print("\n" + "-"*40 + "\n2️⃣ PIPELINE CLASSIFICATION\n" + "-"*40)
    target_clf = 'price_category'
    cols_to_drop_clf = ['prix_mad', 'log_prix', target_clf]
    
    # Enlever les lignes où la catégorie est nulle (au cas où)
    train_df_clf = train_df.dropna(subset=[target_clf])
    test_df_clf = test_df.dropna(subset=[target_clf])
    
    X_train_clf = train_df_clf.drop(columns=cols_to_drop_clf, errors='ignore')
    X_test_clf = test_df_clf.drop(columns=cols_to_drop_clf, errors='ignore')
    
    X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = prepare_data(
        X_train_clf, X_test_clf, train_df_clf[target_clf], test_df_clf[target_clf], "classification"
    )
    train_classification(X_tr_clf, y_tr_clf, X_te_clf, y_te_clf)

    print("\n🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS ! 🎉")

if __name__ == "__main__":
    run_pipeline()