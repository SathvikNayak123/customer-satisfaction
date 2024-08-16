from pipelines.training_pipeline import train_pipeline
from components.stage01_data_ingestion import ingest_data
from components.stage02_data_clean import clean_data
from components.stage03_model_trainer import train_model
from components.stage04_model_evaluation import evaluation
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    training = train_pipeline(
        ingest_data(),
        clean_data(),
        train_model(),
        evaluation(),
    )

    training.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )