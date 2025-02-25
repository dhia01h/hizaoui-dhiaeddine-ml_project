import mlflow
import mlflow.sklearn

# Initialiser le client MLflow
client = mlflow.MlflowClient()

# Nom du modèle
model_name = "ChurnPredictionModel"

# Récupérer toutes les versions du modèle
model_versions = client.search_model_versions(f"name='{model_name}'")

# Stocker les métriques des modèles
models_with_metrics = []

# Récupérer l'Accuracy pour chaque version
for version in model_versions:
    run_id = version.run_id
    metrics = client.get_run(run_id).data.metrics

    if "Accuracy" in metrics:
        accuracy = metrics["Accuracy"]
        models_with_metrics.append({
            "version": version.version,
            "accuracy": accuracy
        })

# Trier les modèles par Accuracy (du plus grand au plus petit)
models_with_metrics.sort(key=lambda x: x["accuracy"], reverse=True)

# Assigner les stages :
for idx, model in enumerate(models_with_metrics):
    version_number = model["version"]
    accuracy = model["accuracy"]

    if idx == 0:
        new_stage = "Production"  # 🚀 Meilleur modèle en Production
    elif idx == 1:
        new_stage = "Staging"  # ✅ Deuxième modèle en Staging
    else:
        new_stage = "Archived"  # 📦 Autres versions archivées

    print(f"🔄 Transition de la version {version_number} (Accuracy: {accuracy}) vers {new_stage}")

    # Mettre à jour le stage du modèle
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage=new_stage
    )

print("✅ Mise à jour des versions terminée !")
