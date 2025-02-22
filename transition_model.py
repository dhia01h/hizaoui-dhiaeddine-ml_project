import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
MODEL_NAME = "ChurnPredictionModel"

def promote_best_model():
    """
    Sélectionne et promeut automatiquement la meilleure version du modèle vers Staging.
    """
    best_model = None
    best_auc = 0

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics
        
        if "roc_auc" in metrics and metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = version

    if best_model:
        print(f"🚀 Promotion du modèle version {best_model.version} en Staging (ROC-AUC: {best_auc})")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=best_model.version,
            stage="Staging"
        )
    else:
        print("❌ Aucun modèle trouvé pour la promotion en Staging.")

if __name__ == "__main__":
    promote_best_model()
