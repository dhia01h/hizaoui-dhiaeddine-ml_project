import argparse
import mlflow
from training.model_training import train_model
from evaluation.model_evaluation import evaluate_model
from data_processing.data_preparation import prepare_data
from utils.mlflow_utils import get_elasticsearch_client, log_to_elasticsearch, init_mlflow
from utils.config import DATASET_PATH

experiment_id = init_mlflow()
es = get_elasticsearch_client()

def main():
    parser = argparse.ArgumentParser(description="Pipeline Churn avec MLflow")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")

    args = parser.parse_args()
    X_train, X_test, y_train, y_test, _ = prepare_data(DATASET_PATH)
    model = None  

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        if args.train:
            model = train_model(X_train, y_train)
        
        if args.evaluate and model:
            metrics = evaluate_model(model, X_test, y_test)
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
                log_to_elasticsearch(es, run_id, metric, value)

if __name__ == "__main__":
    main()
