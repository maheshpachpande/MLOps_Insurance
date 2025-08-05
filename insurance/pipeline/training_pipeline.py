from insurance.entity.config_entity import (TrainingPipelineConfig,
                                            DataIngestionConfig,
                                            DataValidationConfig)

from insurance.entity.artifact_entity import (DataIngestionArtifact,
                                              DataValidationArtifact)

from insurance.components.data_ingestion import DataIngestion
from insurance.components.data_validation import DataValidation
from insurance.exception import CustomException
from insurance.logger import logging



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
        
        
    def start_data_validation(self) -> DataValidationArtifact:
        try:
            data_ingestion_config = DataIngestionConfig()
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=data_ingestion_config.training_file_path,
                                                            test_file_path=data_ingestion_config.testing_file_path)

            data_validation_config = DataValidationConfig()
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            
            logging.info("✅ Data validation completed.")
            return data_validation_artifact
            
        except Exception as e:
            logging.error("❌ Error during data validation.", exc_info=True)
            raise CustomException(e)
            
        

    def run_pipeline(self):
        try:
            logging.info("🚀 Starting training pipeline.")
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact: DataValidationArtifact = self.start_data_validation()
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
