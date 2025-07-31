import os
from pathlib import Path
import logging
from typing import List

# ---------------------------
# ✅ Logging Configuration
# ---------------------------
# This sets up a global logger for the script.
# It logs INFO-level messages and above, with timestamps and levels in each log line.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

# ---------------------------
# ✅ Constants
# ---------------------------
# The main project directory name
PROJECT_NAME = "insurance"

# A list of all files and paths that we want to create for the initial project setup
# These include Python module files, configuration files, Dockerfile, etc.
FILES_TO_CREATE: List[str] = [
    f"{PROJECT_NAME}/__init__.py",                              # Makes us_visa a Python package
    f"{PROJECT_NAME}/components/__init__.py",                   # Package init for components
    f"{PROJECT_NAME}/components/data_ingestion.py",             # Component to ingest data
    f"{PROJECT_NAME}/components/data_validation.py",            # Component to validate input data
    f"{PROJECT_NAME}/components/data_transformation.py",        # Component to transform features
    f"{PROJECT_NAME}/components/model_trainer.py",              # Component to train ML models
    f"{PROJECT_NAME}/components/model_evaluation.py",           # Component to evaluate models
    f"{PROJECT_NAME}/components/model_pusher.py",               # Component to push models to storage
    f"{PROJECT_NAME}/configuration/__init__.py",                # Project configuration logic
    f"{PROJECT_NAME}/constants/__init__.py",                    # Constant values shared across modules
    f"{PROJECT_NAME}/entity/__init__.py",                       # Schema for data/config entities
    f"{PROJECT_NAME}/entity/config_entity.py",                  # Config dataclass definitions
    f"{PROJECT_NAME}/entity/artifact_entity.py",                # Artifact dataclass definitions
    f"{PROJECT_NAME}/exception/__init__.py",                    # Custom exception handling logic
    f"{PROJECT_NAME}/logger/__init__.py",                       # Custom logging (file, level, etc.)
    f"{PROJECT_NAME}/pipeline/__init__.py",                     # Fixed typo: was "pipline"
    f"{PROJECT_NAME}/pipeline/training_pipeline.py",            # Entrypoint to training pipeline
    f"{PROJECT_NAME}/pipeline/prediction_pipeline.py",          # Entrypoint to prediction pipeline
    f"{PROJECT_NAME}/utils/__init__.py",                        # Utility module init
    f"{PROJECT_NAME}/utils/main_utils.py",                      # Helper functions used across project
    "app.py",                                                   # Flask or FastAPI serving script
    "requirements.txt",                                         # Python dependencies
    "Dockerfile",                                               # Docker container specification
    ".dockerignore",                                            # Ignore rules for Docker builds
    "demo.py",                                                  # Quick demo/test script
    "setup.py",                                                 # Package installer
    "config/model.yaml",                                        # Model-related config
    "config/schema.yaml",                                       # Schema/validation config
]

# ---------------------------
# ✅ Function to Scaffold Project
# ---------------------------
def scaffold_project(file_paths: List[str]) -> None:
    """
    Creates all directories and files specified in file_paths.
    Ensures the directory exists before creating files.
    Logs each action taken.
    """
    for filepath_str in file_paths:
        filepath = Path(filepath_str)                      # Convert string to Path object
        filedir, filename = os.path.split(filepath)        # Separate directory and file name

        try:
            # Create the directory if it doesn't exist
            if filedir:
                os.makedirs(filedir, exist_ok=True)
                logging.info(f"Created directory: {filedir}")

            # Create the file only if it doesn't exist or is empty
            if not filepath.exists() or filepath.stat().st_size == 0:
                filepath.touch()  # Create an empty file
                logging.info(f"Created file: {filepath}")
            else:
                logging.info(f"File already exists at: {filepath}")
        except Exception as e:
            # Log any error that occurs during file or directory creation
            logging.error(f"Error while creating {filepath}: {e}")

# ---------------------------
# ✅ Entry Point
# ---------------------------
# This ensures the script only runs when called directly
# Useful when this script is imported into other modules (it won't auto-run)
if __name__ == "__main__":
    scaffold_project(FILES_TO_CREATE)