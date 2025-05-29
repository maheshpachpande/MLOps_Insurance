from src.insurance_fraud_detection.logger import logging
from src.insurance_fraud_detection.exception import CustomException
from src.insurance_fraud_detection.components.data_ingestion import DataIngestion
import os
import sys



if __name__ == "__main__":
    logging.info("Starting the Insurance Fraud Detection Pipeline")
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
        logging.info("Insurance Fraud Detection Pipeline completed successfully")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise CustomException(e, sys)


