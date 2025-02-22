import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.config import DATASET_PATH

def prepare_data(filepath=DATASET_PATH):
    """Charge et prétraite les données"""
    print(f"📂 Chargement des données depuis {filepath}...")
    df = pd.read_csv(filepath)

    if "Churn" not in df.columns:
        raise ValueError("❌ Erreur : La colonne 'Churn' est absente du dataset.")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("✅ Données préparées avec succès !")
    return X_train, X_test, y_train, y_test, scaler
