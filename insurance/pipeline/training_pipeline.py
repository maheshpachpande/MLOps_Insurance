from insurance.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
)

from insurance.entity.artifact_entity import DataIngestionArtifact
from insurance.components.data_ingestion import DataIngestion
from insurance.exception import CustomException
from insurance.logger import logging

import os
import sys


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig()
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("✅ Data ingestion completed.")
            return data_ingestion_artifact

        except Exception as e:
            logging.error("❌ Error during data ingestion.", exc_info=True)
            raise CustomException(e)

    def run_pipeline(self):
        try:
            logging.info("🚀 Starting training pipeline.")
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            logging.info("✅ Pipeline executed successfully.")
        except Exception as e:
            logging.error("❌ Pipeline execution failed.", exc_info=True)
            raise CustomException(e)


# 🚦 Entry point
if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.critical("🔥 Failed to run the TrainingPipeline from __main__", exc_info=True)
        raise e
