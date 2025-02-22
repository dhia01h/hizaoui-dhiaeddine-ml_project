import pytest
import numpy as np
from data_processing.data_preparation import prepare_data
from training.model_training import train_model
from evaluation.model_evaluation import evaluate_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def test_evaluate_model():
    """Teste l'évaluation du modèle avec des données correctes"""
    X_train, X_test, y_train, y_test, _ = prepare_data()
    model = train_model(X_train, y_train)

    # Assurer que y_test contient au moins une classe positive et une classe négative
    if len(np.unique(y_test)) == 1:
        y_test[0] = 1 - y_test[0]  # Modifier un élément pour éviter le warning Sklearn

    metrics = evaluate_model(model, X_test, y_test)

    # Vérifications des métriques
    assert 0 <= metrics["Accuracy"] <= 1, "Accuracy hors limites"
    assert 0 <= metrics["Precision"] <= 1, "Precision hors limites"
    assert 0 <= metrics["Recall"] <= 1, "Recall hors limites"
    assert 0 <= metrics["F1-score"] <= 1, "F1-score hors limites"
    assert 0 <= metrics["ROC-AUC"] <= 1, "ROC-AUC hors limites"

    print("✅ Test d'évaluation du modèle réussi !")

def test_train_model():
    """Teste l'entraînement du modèle"""
    X_train, _, y_train, _, _ = prepare_data()
    model = train_model(X_train, y_train)

    # Vérifier que le modèle est bien entraîné
    assert model is not None, "Le modèle n'a pas été entraîné"
    assert hasattr(model, "predict"), "Le modèle ne possède pas de méthode predict"

    print("✅ Test d'entraînement du modèle réussi !")

def test_prepare_data():
    """Teste la préparation des données"""
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # Vérifications sur les dimensions
    assert X_train.shape[0] > 0, "X_train est vide"
    assert X_test.shape[0] > 0, "X_test est vide"
    assert y_train.shape[0] > 0, "y_train est vide"
    assert y_test.shape[0] > 0, "y_test est vide"

    # Vérification de la normalisation
    assert np.allclose(X_train.mean(), 0, atol=1), "Les données ne semblent pas être normalisées"
    assert np.allclose(X_test.mean(), 0, atol=1), "Les données ne semblent pas être normalisées"

    print("✅ Test de préparation des données réussi !")
