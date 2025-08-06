from insurance.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)


import sys
from insurance.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)

from insurance.components.data_ingestion import DataIngestion
from insurance.components.data_validation import DataValidation
from insurance.components.data_transformation import DataTransformation
from insurance.components.model_trainer import ModelTrainer
from insurance.components.model_evaluation import ModelEvaluation
from insurance.components.model_pusher import ModelPusher

from insurance.exception import CustomException
from insurance.logger import logging
from insurance.utils.main_utils import read_yaml_file


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig()
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("✅ Data ingestion completed.")
            return data_ingestion_artifact

        except Exception as e:
            logging.error("❌ Error during data ingestion.", exc_info=True)
            raise CustomException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
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
            raise CustomException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig()
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info("✅ Data transformation completed.")
            return data_transformation_artifact

        except Exception as e:
            logging.error("❌ Error during data transformation.", exc_info=True)
            raise CustomException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig()
            trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = trainer.initiate_model_trainer()

            logging.info("✅ Model training completed.")
            return model_trainer_artifact

        except Exception as e:
            logging.error("❌ Error during model training.", exc_info=True)
            raise CustomException(e, sys)


    
    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact,
                                ):
        try:
            model_eval_config = ModelEvaluationConfig()
            model_eval = ModelEvaluation(model_eval_config, data_validation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except  Exception as e:
            raise  CustomException(e, sys)
    
    
    def start_model_pusher(self, model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig()
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except  Exception as e:
            raise  CustomException(e, sys)
    
    
    def run_pipeline(self):
        try:
            logging.info("🚀 Starting training pipeline.")

            # Step-by-step execution
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)

            logging.info("✅ Training pipeline completed successfully.")

        except Exception as e:
            logging.error("❌ Pipeline execution failed.", exc_info=True)
            raise CustomException(e, sys)


# 🚦 Entry point
if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.critical("🔥 Failed to run the TrainingPipeline from __main__", exc_info=True)
        raise e
