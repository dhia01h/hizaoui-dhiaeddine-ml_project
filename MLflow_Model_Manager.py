import mlflow
from mlflow.tracking import MlflowClient

# 🔹 Nom du modèle dans MLflow Model Registry
MODEL_NAME = "ChurnPredictionModel"

# 🔹 Initialisation du client MLflow
client = MlflowClient()

# 🔹 Récupérer toutes les versions du modèle
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# 🎯 Filtrer les modèles en Staging
staging_models = [v for v in versions if v.current_stage == "Staging"]

if not staging_models:
    print("❌ Aucun modèle trouvé pour la promotion en Staging.")
else:
    # 📌 Sélectionner le modèle avec la version la plus récente
    best_model = max(staging_models, key=lambda v: int(v.version))
    
    print(f"✅ Promotion du modèle {MODEL_NAME} version {best_model.version} en Production...")

    # 🔥 Promouvoir le modèle en Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=best_model.version,
        stage="Production"
    )
    
    print(f"🚀 Modèle version {best_model.version} promu en Production !")

    # 🗄️ Archivage des anciens modèles en Production
    production_models = [v for v in versions if v.current_stage == "Production" and v.version != best_model.version]

    for model in production_models:
        print(f"📦 Archivage de l'ancien modèle version {model.version}...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model.version,
            stage="Archived"
        )
        print(f"📌 Modèle version {model.version} archivé.")
