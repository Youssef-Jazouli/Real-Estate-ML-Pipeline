import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_engineer(df):
    print("⏳ Séparation Train/Test et Feature Engineering...")
    
    # 1. Nettoyage initial (Enlever les colonnes inutiles pour le ML)
    cols_to_drop = ['titre', 'prix_par_m2'] # prix_par_m2 enlevé pour éviter la fuite de données
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 2. Séparation (80% Train, 20% Test) AVANT les transformations
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    def apply_features(data):
        df_copy = data.copy()
        
        # Log transformation du prix
        if 'prix_mad' in df_copy.columns:
            df_copy['log_prix'] = np.log1p(df_copy['prix_mad'])
            
            # Créer une cible pour la classification (Catégorie de prix)
            bins = [0, 500000, 1500000, float('inf')]
            labels = ['Économique', 'Moyen', 'Luxe']
            df_copy['price_category'] = pd.cut(df_copy['prix_mad'], bins=bins, labels=labels)
            
        # Interaction (Surface totale des pièces)
        if 'surface_m2' in df_copy.columns and 'nb_chambres' in df_copy.columns:
            df_copy['surface_par_chambre'] = df_copy['surface_m2'] / (df_copy['nb_chambres'] + 1)
            
        # Variables temporelles
        if 'scraped_at' in df_copy.columns:
            df_copy['scraped_at'] = pd.to_datetime(df_copy['scraped_at'])
            df_copy['mois_annonce'] = df_copy['scraped_at'].dt.month
            df_copy = df_copy.drop(columns=['scraped_at'])
            
        return df_copy

    # 3. Appliquer les transformations
    train_eng = apply_features(train_df)
    test_eng = apply_features(test_df)
    
    os.makedirs("data/processed", exist_ok=True)
    train_eng.to_csv("data/processed/train_eng.csv", index=False)
    test_eng.to_csv("data/processed/test_eng.csv", index=False)
    
    print("✅ Feature Engineering terminé.")
    return train_eng, test_eng