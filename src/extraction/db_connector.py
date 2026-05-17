import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def extract_obt():
    try:
        print("⏳ Extraction des données depuis ml_schema.feature_store...")
        
        # Connexion à la base de données
        url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(url)
        
        # Extraire la table OBT (Data Engineering Output)
        query = "SELECT * FROM ml_schema.feature_store"
        df = pd.read_sql(query, engine)
        
        # Sauvegarde
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/obt_data.csv", index=False)
        
        print(f"✅ Extraction réussie ! Taille: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise