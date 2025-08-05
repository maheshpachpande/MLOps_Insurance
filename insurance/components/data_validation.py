import os
import sys
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from insurance.constants.training_pipeline import SCHEMA_FILE_PATH
from insurance.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from insurance.entity.config_entity import DataIngestionConfig, DataValidationConfig
from insurance.exception import CustomException
from insurance.logger import logging
from insurance.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e)

    def drop_zero_std_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.loc[:, dataframe.std() > 0]

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = len(self._schema_config["columns"])
            actual_columns = len(dataframe.columns)
            logging.info(f"Expected columns: {expected_columns}, Found: {actual_columns}")
            return expected_columns == actual_columns
        except Exception as e:
            raise CustomException(e)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            missing_numerical = [col for col in numerical_columns if col not in dataframe.columns]
            if missing_numerical:
                logging.warning(f"Missing numerical columns: {missing_numerical}")
                return False
            return True
        except Exception as e:
            raise CustomException(e)

    def is_categorical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            categorical_columns = self._schema_config["categorical_columns"]
            missing_categorical = [col for col in categorical_columns if col not in dataframe.columns]
            if missing_categorical:
                logging.warning(f"Missing categorical columns: {missing_categorical}")
                return False
            return True
        except Exception as e:
            raise CustomException(e)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                if d1.dtype == "object" or d2.dtype == "object":
                    d1 = d1.astype(str)
                    d2 = d2.astype(str)
                    label_encoder = LabelEncoder()
                    combined = pd.concat([d1, d2])
                    label_encoder.fit(combined)
                    d1 = label_encoder.transform(d1)
                    d2 = label_encoder.transform(d2)

                statistic, p_value = ks_2samp(d1, d2)
                p_value = float(p_value)
                drift_status = bool(p_value < threshold)

                if drift_status:
                    status = False

                report[column] = {
                    "p_value": p_value,
                    "drift_status": drift_status
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        except Exception as e:
            raise CustomException(e)

    def detect_prior_probability_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str) -> dict:
        try:
            base_dist = base_df[target_col].value_counts(normalize=True)
            current_dist = current_df[target_col].value_counts(normalize=True)

            all_classes = set(base_dist.index).union(current_dist.index)
            drift_report = {}

            for cls in all_classes:
                base_p = float(base_dist.get(cls, 0.0))
                current_p = float(current_dist.get(cls, 0.0))
                diff = abs(base_p - current_p)

                drift_report[str(cls)] = {
                    "base_probability": base_p,
                    "current_probability": current_p,
                    "absolute_difference": diff
                }

            prior_drift_path = self.data_validation_config.prior_drift_report_file_path
            os.makedirs(os.path.dirname(prior_drift_path), exist_ok=True)
            write_yaml_file(file_path=prior_drift_path, content=drift_report)

            return drift_report
        except Exception as e:
            raise CustomException(e)

    def detect_concept_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str) -> float:
        try:
            if isinstance(target_col, (list, tuple)):
                target_col = target_col[0]

            label_encoder = LabelEncoder()
            base_df[target_col] = label_encoder.fit_transform(base_df[target_col])
            current_df[target_col] = label_encoder.transform(current_df[target_col])

            X_train = base_df.drop(columns=[target_col])
            y_train = base_df[target_col]
            X_test = current_df.drop(columns=[target_col])
            y_test = current_df[target_col]

            # Encode categorical features
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_test[col].dtype == 'object':
                    enc = LabelEncoder()
                    all_values = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
                    enc.fit(all_values)
                    X_train[col] = enc.transform(X_train[col].astype(str))
                    X_test[col] = enc.transform(X_test[col].astype(str))

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            return accuracy
        except Exception as e:
            raise CustomException(e)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""

            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            if not self.validate_number_of_columns(train_df):
                error_message += "Train data: incorrect number of columns.\n"
            if not self.validate_number_of_columns(test_df):
                error_message += "Test data: incorrect number of columns.\n"

            if not self.is_numerical_column_exist(train_df):
                error_message += "Train data: missing numerical columns.\n"
            if not self.is_numerical_column_exist(test_df):
                error_message += "Test data: missing numerical columns.\n"

            if not self.is_categorical_column_exist(train_df):
                error_message += "Train data: missing categorical columns.\n"
            if not self.is_categorical_column_exist(test_df):
                error_message += "Test data: missing categorical columns.\n"

            if error_message:
                raise Exception(error_message)

            # Dataset Drift / Covariate Shift
            drift_status = self.detect_dataset_drift(
                train_df.drop(columns=[self._schema_config["target_column"][0]]),
                test_df.drop(columns=[self._schema_config["target_column"][0]])
            )

            # Prior Probability Drift
            prior_drift = self.detect_prior_probability_drift(
                train_df, test_df, self._schema_config["target_column"][0]
            )

            # Concept Drift
            concept_accuracy = self.detect_concept_drift(
                train_df.copy(), test_df.copy(), self._schema_config["target_column"]
            )
            logging.info(f"Concept Drift Accuracy on test: {concept_accuracy:.4f}")

            if concept_accuracy < 0.7:
                logging.warning("⚠️ Possible Concept Drift: accuracy below threshold (0.7)")

            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=train_file_path,
                valid_test_file_path=test_file_path,
                invalid_train_file_path="",
                invalid_test_file_path="",
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"✅ Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion_artifact = DataIngestionArtifact(
        trained_file_path=data_ingestion_config.training_file_path,
        test_file_path=data_ingestion_config.testing_file_path
    )

    data_validation_config = DataValidationConfig()
    data_validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=data_validation_config
    )
    data_validation.initiate_data_validation()
