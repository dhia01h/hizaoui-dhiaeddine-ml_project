from mlflow.tracking import MlflowClient

client = MlflowClient()
MODEL_NAME = "ChurnPredictionModel"

# Récupérer toutes les versions du modèle
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

print("\n🔎 Liste des modèles enregistrés :")
for v in versions:
    print(f"🔹 Version: {v.version}, Stage: {v.current_stage}")

