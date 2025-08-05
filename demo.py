from insurance.logger import logging
from insurance.exception import CustomException

from insurance.pipeline.training_pipeline import TrainingPipeline


try:
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()
except Exception as e:
    logging.error("❌ Pipeline execution failed.", exc_info=True)
    raise CustomException(e)