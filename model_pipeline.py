import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# 🔹 Préparation des données
def prepare_data(filepath):
    print(f"📂 Chargement des données depuis {filepath}...")
    df = pd.read_csv(filepath)

    # Vérifier si la colonne cible "Churn" existe
    if "Churn" not in df.columns:
        raise ValueError("❌ Erreur : La colonne 'Churn' est absente du dataset.")

    # Encodage des variables catégoriques
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Séparation des features et de la cible
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("✅ Données préparées avec succès !")
    return X_train, X_test, y_train, y_test, scaler


# 🔹 Entraînement du modèle
def train_model(X_train, y_train):
    print("🚀 Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès !")
    return model


# 🔹 Évaluation du modèle
def evaluate_model(model, X_test, y_test):
    print("📊 Évaluation du modèle en cours...")
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred),
    }

    print("✅ Évaluation terminée.")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


# 🔹 Sauvegarde du modèle
def save_model(model, filename="random_forest.pkl"):
    if model is None:
        print("❌ Erreur : Aucun modèle à sauvegarder.")
        return
    joblib.dump(model, filename)
    print(f"💾 Modèle sauvegardé sous {filename}")


# 🔹 Chargement du modèle
def load_model(filename="random_forest.pkl"):
    try:
        print(f"🔄 Chargement du modèle depuis {filename}...")
        model = joblib.load(filename)
        print("✅ Modèle chargé avec succès !")
        return model
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier {filename} introuvable.")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None
