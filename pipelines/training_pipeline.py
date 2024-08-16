from zenml import pipeline
from components.stage01_data_ingestion import ingest_data
from components.stage02_data_clean import clean_data
from components.stage03_model_trainer import train_model
from components.stage04_model_evaluation import evaluate_model

@pipeline
def training_pipeline():
    df=ingest_data()
    clean_data(df)
    train_model(df)
    evaluate_model(df)
    