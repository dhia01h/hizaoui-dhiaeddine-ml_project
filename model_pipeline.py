"""
Main script for the Churn Prediction pipeline.
Handles data loading, training, evaluation, and MLflow logging.
"""


import pandas as pd
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


def prepare_data(filepath):
    """
    Charge et prépare les données à partir du fichier CSV.

    Args:
        filepath (str): Chemin vers le fichier CSV contenant les données.

    Returns:
        tuple: (x_train, x_test, y_train, y_test, scaler) après transformation.
    """
    print(f"📂 Chargement des données depuis {filepath}...")
    df = pd.read_csv(filepath)

    # Vérifier si la colonne cible "Churn" existe
    if "Churn" not in df.columns:
        raise ValueError("❌ Erreur : La colonne 'Churn' est absente du dataset.")

    # Encodage des variables catégoriques
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Séparation des features et de la cible
    x_features = df.drop(columns=["Churn"])
    y_target = df["Churn"]

    # Normalisation des données
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_features)

    # Split des données
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_target, test_size=0.2, random_state=42
    )

    print("✅ Données préparées avec succès !")
    return x_train, x_test, y_train, y_test, scaler


def train_model(x_train, y_train):
    """
    Entraîne un modèle Random Forest.

    Args:
        x_train (array-like): Données d'entraînement.
        y_train (array-like): Labels d'entraînement.

    Returns:
        RandomForestClassifier: Modèle entraîné.
    """
    print("🚀 Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    print("✅ Modèle entraîné avec succès !")
    return model


def evaluate_model(model, x_test, y_test):
    """
    Évalue un modèle sur les données de test.

    Args:
        model (RandomForestClassifier): Modèle à évaluer.
        x_test (array-like): Données de test.
        y_test (array-like): Labels de test.

    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation.
    """
    print("📊 Évaluation du modèle en cours...")
    y_pred = model.predict(x_test)

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


def save_model(model, filename="random_forest.pkl"):
    """
    Sauvegarde un modèle entraîné sur disque.

    Args:
        model (RandomForestClassifier): Modèle à sauvegarder.
        filename (str): Nom du fichier où enregistrer le modèle.
    """
    if model is None:
        print("❌ Erreur : Aucun modèle à sauvegarder.")
        return
    joblib.dump(model, filename)
    print(f"💾 Modèle sauvegardé sous {filename}")


def load_model(filename="random_forest.pkl"):
    """
    Charge un modèle sauvegardé depuis le disque.

    Args:
        filename (str): Nom du fichier contenant le modèle sauvegardé.

    Returns:
        RandomForestClassifier or None: Le modèle chargé ou None si échec.
    """
    try:
        print(f"🔄 Chargement du modèle depuis {filename}...")
        model = joblib.load(filename)
        print("✅ Modèle chargé avec succès !")
        return model
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier {filename} introuvable.")
        return None
    except ValueError as error:
        print(f"❌ Erreur lors du chargement du modèle : {error}")
        return None
