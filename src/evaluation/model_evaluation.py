from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle"""
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
