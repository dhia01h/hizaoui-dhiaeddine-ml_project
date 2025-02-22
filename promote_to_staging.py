from mlflow.tracking import MlflowClient

client = MlflowClient()
MODEL_NAME = "ChurnPredictionModel"
MODEL_VERSION = "9"  # Remplace par la version que tu veux promouvoir

# Promouvoir en Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    stage="Staging"
)

print(f"✅ Modèle version {MODEL_VERSION} promu en Staging.")

