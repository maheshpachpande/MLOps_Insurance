
from dataclasses import dataclass
import os
from insurance.constants  import training_pipeline

from insurance.constants.training_pipeline import *



@dataclass
class TrainingPipelineConfig:
        pipeline_name: str = training_pipeline.PIPELINE_NAME
        artifact_dir: str = training_pipeline.ARTIFACT_DIR
        
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()



@dataclass
class DataIngestionConfig:
        
        data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
        train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        collection_name:str = DATA_INGESTION_COLLECTION_NAME



@dataclass
class DataValidationConfig:
        
        data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
        valid_data_dir: str = os.path.join(data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        invalid_data_dir: str = os.path.join(data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        valid_train_file_path: str = os.path.join(valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        valid_test_file_path: str = os.path.join(valid_data_dir, training_pipeline.TEST_FILE_NAME)
        invalid_train_file_path: str = os.path.join(invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        invalid_test_file_path: str = os.path.join(invalid_data_dir, training_pipeline.TEST_FILE_NAME)
        prior_drift_report_file_path: str = os.path.join(data_validation_dir,
                                                training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                training_pipeline.PRIOR_DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,)
        drift_report_file_path: str = os.path.join(data_validation_dir,
                                                training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,)


@dataclass
class DataTransformationConfig:
        data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME )
        transformed_train_file_path: str = os.path.join(  data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),)
        transformed_test_file_path: str = os.path.join( data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        training_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        transformed_object_file_path: str = os.path.join(  data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        training_pipeline.PREPROCSSING_OBJECT_FILE_NAME,)



@dataclass
class ModelTrainerConfig:
        model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        trained_model_file_path: str = os.path.join(
        model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
        training_pipeline.MODEL_FILE_NAME
        )
        expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        artifact_yaml_path = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME, MODEL_ARTIFACT_FILE_NAME)


@dataclass
class ModelEvaluationConfig:
        model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, training_pipeline.MODEL_EVALUATION_DIR_NAME
        )
        report_file_path = os.path.join( model_evaluation_dir,training_pipeline.MODEL_EVALUATION_REPORT_NAME)
        change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

