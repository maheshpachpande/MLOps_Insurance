import logging
import os
from datetime import datetime # to get current time

# This module sets up logging for the insurance fraud detection application.
# It creates a log file with a timestamp in the 'logs' directory.
# If the 'logs' directory does not exist, it will be created.
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

# Create the full path for the log file
LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

# Configure the logging settings and specify the log file path.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)