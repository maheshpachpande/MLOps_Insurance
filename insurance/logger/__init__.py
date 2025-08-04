import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Simple path setup
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Test message
logger.info("✅ This should appear in both console and log file.")
