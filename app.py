from src.insurance_fraud_detection.logger import logging
from src.insurance_fraud_detection.exception import CustomException
import sys
import os

logging.info("Logging has been set up successfully.")

try:
    # Simulating some code execution
    logging.info("Starting the application...")
    
    # Simulate a division by zero error
    result = 10 / 0
    logging.info(f"Result: {result}")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    raise CustomException(e, sys) from e

