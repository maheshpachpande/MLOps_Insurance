import os, sys
import numpy as np
from typing import Dict, Any
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from insurance.exception import CustomException
from insurance.logger import logging
from insurance.entity.artifact_entity import (DataTransformationArtifact, 
                                              ModelTrainerArtifact
)
from insurance.entity.config_entity import (DataTransformationConfig, 
                                            ModelTrainerConfig
)
from insurance.ml.metric.classification_metrics import get_classification_score
from insurance.ml.model.estimator import InsuranceModel

from insurance.utils.main_utils import (load_numpy_array_data, 
                                        save_object, 
                                        load_object,
                                        write_yaml_file)

from dataclasses import asdict
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')





class ModelTrainer:

    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def perform_hyperparameter_tuning(
        self, x_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
        try:
            models: Dict[str, Any] = {
                "LogisticRegression": {
                    "model": LogisticRegression(solver='liblinear'),
                    "params": {
                        "C": [0.1, 1, 10],
                        "penalty": ["l1", "l2"]
                    }
                },
                "RandomForestClassifier": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20]
                    }
                },
                "XGBClassifier": {
                    "model": XGBClassifier(eval_metric='logloss'),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1]
                    }
                }
            }

            best_score = -np.inf
            best_model = None
            best_model_name = None
            best_params = None

            for name, config in models.items():
                logging.info(f"Tuning hyperparameters for {name}...")
                grid_search = GridSearchCV(
                    estimator=config["model"],
                    param_grid=config["params"],
                    scoring="f1_macro",
                    cv=3,
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(x_train, y_train)
                logging.info(f"{name} best F1_macro score: {grid_search.best_score_}")
                logging.info(f"{name} best parameters: {grid_search.best_params_}")

                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_model_name = name
                    best_params = grid_search.best_params_

            if best_model is None:
                raise CustomException("No best model found during hyperparameter tuning.", sys)

            logging.info(f"✅ Best model selected: {best_model_name}")
            logging.info(f"✅ Best hyperparameters: {best_params}")

            return best_model

        except Exception as e:
            raise CustomException(e, sys)


    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
        try:
            return self.perform_hyperparameter_tuning(x_train, y_train)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed datasets...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Encode string labels to numeric
            if y_train.dtype.kind in {'U', 'S', 'O'}:
                logging.info("Encoding labels with LabelEncoder...")
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

            model = self.train_model(x_train, y_train)

            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            if classification_train_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise CustomException("Trained model did not meet expected F1 threshold.", sys)

            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            overfit_gap = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if overfit_gap > self.model_trainer_config.overfitting_underfitting_threshold:
                raise CustomException(f"Model overfitting detected. Gap: {overfit_gap}", sys)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            insurance_model = InsuranceModel(preprocessor=preprocessor, model=model)

            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=insurance_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            # Write yaml file
            write_yaml_file(file_path=self.model_trainer_config.artifact_yaml_path, content=asdict(model_trainer_artifact))

            logging.info(f"Model trainer completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        data_transformation_config = DataTransformationConfig()
        data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path=data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=data_transformation_config.transformed_test_file_path,
            transformed_object_file_path=data_transformation_config.transformed_object_file_path
        )

        model_trainer_config = ModelTrainerConfig()
        trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        trainer.initiate_model_trainer()
    except Exception as e:
        raise CustomException(e, sys)
