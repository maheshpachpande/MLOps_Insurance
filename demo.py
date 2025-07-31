from insurance.logger import logging
from insurance.exception import CustomException

logging.info("Logging has started")

try:
    a = 1/0
except Exception as e:
    raise CustomException(e)