"""
Main script for churn prediction pipeline with MLflow and Elasticsearch.
"""

import os
import argparse
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
from mlflow_utils import get_elasticsearch_client, log_to_elasticsearch

# 📌 Configuration de MLflow
TRACKING_URI = "http://localhost:5000"  # ✅ Vérifie bien que MLflow tourne sur ce port
EXPERIMENT_NAME = "Churn Prediction Experiment"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# 📌 Vérifier si l'expérience MLflow existe, sinon la créer
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"⚠️ Expérience '{EXPERIMENT_NAME}' non trouvée, création en cours...")
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
    print(f"✅ Expérience créée avec ID : {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"🔄 Expérience existante trouvée : ID {experiment_id}")

mlflow.set_experiment(EXPERIMENT_NAME)

# 📌 Connexion à Elasticsearch
es = get_elasticsearch_client()
if es is None:
    print("⚠️ Elasticsearch est injoignable, les logs ne seront pas envoyés.")


def update_model_stages(model_name="ChurnPredictionModel"):
    """
    Met à jour les stages des modèles en fonction de l'accuracy.

    - Le modèle avec la meilleure accuracy passe en "Production".
    - L'ancien modèle en "Production" est archivé.
    - Les autres modèles sont en "Staging".

    Args:
        model_name (str): Nom du modèle dans MLflow Model Registry.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    best_version = None
    best_accuracy = 0
    model_versions = []

    for v in versions:
        run_id = v.run_id
        metrics = client.get_run(run_id).data.metrics
        accuracy = metrics.get("Accuracy", 0)  # Récupérer l'accuracy

        model_versions.append(
            {"version": v.version, "accuracy": accuracy, "stage": v.current_stage}
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_version = v

    if best_version:
        print(
            f"🔹 La meilleure version est {best_version.version} avec Accuracy: {best_accuracy}"
        )

        # Archiver l'ancien modèle en "Production"
        for v in model_versions:
            if v["stage"] == "Production":
                client.transition_model_version_stage(
                    model_name, v["version"], "Archived"
                )
                client.set_model_version_tag(
                    model_name, v["version"], "stage", "Archived"
                )
                print(f"📌 Version {v['version']} archivée.")

        # Passer la meilleure version en "Production"
        client.transition_model_version_stage(
            model_name, best_version.version, "Production"
        )
        client.set_model_version_tag(
            model_name, best_version.version, "stage", "Production"
        )
        print(f"🚀 Version {best_version.version} mise en Production.")

        # Les autres versions passent en "Staging"
        for v in model_versions:
            if v["version"] != best_version.version and v["stage"] != "Archived":
                client.transition_model_version_stage(
                    model_name, v["version"], "Staging"
                )
                client.set_model_version_tag(
                    model_name, v["version"], "stage", "Staging"
                )
                print(f"⚙️ Version {v['version']} mise en Staging.")
    else:
        print("❌ Aucun modèle avec une accuracy valide trouvé.")


def register_model(run_id, model, x_train):
    """
    Enregistre un modèle dans la Model Registry de MLflow.

    Args:
        run_id (str): ID du run MLflow.
        model (sklearn.base.BaseEstimator): Modèle entraîné.
        x_train (numpy.ndarray): Exemple de données d'entrée pour MLflow.
    """
    model_uri = f"runs:/{run_id}/model"
    input_example = np.array([x_train[0]])

    try:
        mlflow.sklearn.log_model(model, "model", input_example=input_example)
        registered_model = mlflow.register_model(model_uri, "ChurnPredictionModel")

        print(f"✅ Modèle enregistré dans la Model Registry : {registered_model}")

        # 📌 Mettre à jour les stages après l'enregistrement
        update_model_stages("ChurnPredictionModel")

    except ValueError as e:
        print(f"❌ Erreur lors de l'enregistrement du modèle : {e}")


def main():
    """
    Exécute le pipeline de churn avec MLflow.
    """
    parser = argparse.ArgumentParser(description="Pipeline Churn avec MLflow")
    parser.add_argument(
        "--data", type=str, default="churn-bigml-80.csv", help="Chemin du fichier CSV"
    )
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument(
        "--save",
        type=str,
        default="random_forest.pkl",
        help="Nom du fichier pour sauvegarder le modèle",
    )
    parser.add_argument("--load", type=str, help="Charger un modèle sauvegardé")

    args = parser.parse_args()

    print("📂 Chargement des données...")
    x_train, x_test, y_train, y_test, _ = prepare_data(args.data)
    model = None

    if args.load and not os.path.exists(args.load):
        print(f"❌ Fichier modèle '{args.load}' non trouvé. Vérifiez le chemin.")
        return

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_param("data_file", args.data)

        if args.load:
            print(f"🔄 Chargement du modèle depuis {args.load}...")
            model = load_model(args.load)
            if model is None:
                print("❌ Erreur : Impossible de charger le modèle.")
                return
            print("✅ Modèle chargé avec succès !")

        if args.train:
            print("🚀 Entraînement du modèle en cours...")
            model = train_model(x_train, y_train)
            if model is None:
                print("❌ Erreur : L'entraînement du modèle a échoué.")
                return
            print("✅ Modèle entraîné avec succès !")

            register_model(run_id, model, x_train)

            if args.save:
                save_model(model, args.save)
                mlflow.log_artifact(args.save)  # 📂 Ajout du modèle comme artifact
                print(
                    f"💾 Modèle sauvegardé sous {args.save} et ajouté aux artifacts MLflow."
                )

        if args.evaluate:
            if model is None:
                print(
                    "⚠️ Aucun modèle trouvé ! Veuillez en entraîner un ou en charger un."
                )
            else:
                print("📊 Évaluation du modèle en cours...")
                metrics = evaluate_model(model, x_test, y_test)

                metrics_file = "metrics.txt"
                with open(metrics_file, "w") as f:
                    for metric, value in metrics.items():
                        f.write(f"{metric}: {value}\n")

                mlflow.log_artifact(metrics_file)  # 📂 Ajout du fichier aux artifacts

                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)
                    print(f"📤 Enregistrement métrique {metric}: {value} dans MLflow.")

                    try:
                        log_to_elasticsearch(es, run_id, metric, value)
                    except ConnectionError as error:
                        print(
                            f"⚠️ Impossible d'envoyer {metric} à Elasticsearch : {error}"
                        )


if __name__ == "__main__":
    main()
