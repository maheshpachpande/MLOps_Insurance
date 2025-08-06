import os
import sys
import shutil
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from insurance.constants.training_pipeline import SCHEMA_FILE_PATH, OUTPUT_PATH, DATA_VALIDATION_VALIDATED_PATH
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
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        expected_columns = len(self._schema_config["columns"])
        actual_columns = len(dataframe.columns)
        logging.info(f"Expected columns: {expected_columns}, Found: {actual_columns}")
        return expected_columns == actual_columns

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        numerical_columns = self._schema_config["numerical_columns"]
        missing = [col for col in numerical_columns if col not in dataframe.columns]
        if missing:
            logging.warning(f"Missing numerical columns: {missing}")
            return False
        return True

    def is_categorical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        categorical_columns = self._schema_config["categorical_columns"]
        missing = [col for col in categorical_columns if col not in dataframe.columns]
        if missing:
            logging.warning(f"Missing categorical columns: {missing}")
            return False
        return True

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        status = True
        report = {}

        for column in base_df.columns:
            d1 = base_df[column].dropna()
            d2 = current_df[column].dropna()

            if d1.dtype == "object" or d2.dtype == "object":
                le = LabelEncoder()
                combined = pd.concat([d1.astype(str), d2.astype(str)])
                le.fit(combined)
                d1 = le.transform(d1.astype(str))
                d2 = le.transform(d2.astype(str))

            _, p_value = ks_2samp(d1, d2)
            drift_status = p_value < threshold
            report[column] = {"p_value": float(p_value), "drift_status": drift_status}

            if drift_status:
                status = False

        write_yaml_file(self.data_validation_config.drift_report_file_path, content=report)
        return status

    def detect_prior_probability_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str):
        base_dist = base_df[target_col].value_counts(normalize=True)
        curr_dist = current_df[target_col].value_counts(normalize=True)

        all_classes = set(base_dist.index).union(curr_dist.index)
        report = {}

        for cls in all_classes:
            base_p = float(base_dist.get(cls, 0.0))
            curr_p = float(curr_dist.get(cls, 0.0))
            report[str(cls)] = {
                "base_probability": base_p,
                "current_probability": curr_p,
                "absolute_difference": abs(base_p - curr_p)
            }

        write_yaml_file(self.data_validation_config.prior_drift_report_file_path, content=report)
        return report

    def detect_concept_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str) -> float:
        if isinstance(target_col, list):
            target_col = target_col[0]

        le = LabelEncoder()
        base_df[target_col] = le.fit_transform(base_df[target_col])
        current_df[target_col] = le.transform(current_df[target_col])

        X_train = base_df.drop(columns=[target_col])
        y_train = base_df[target_col]
        X_test = current_df.drop(columns=[target_col])
        y_test = current_df[target_col]

        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                enc = LabelEncoder()
                all_vals = pd.concat([X_train[col], X_test[col]]).astype(str)
                enc.fit(all_vals)
                X_train[col] = enc.transform(X_train[col].astype(str))
                X_test[col] = enc.transform(X_test[col].astype(str))

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model = LogisticRegression(max_iter=200000, solver="saga")
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, predictions)
        return float(acc)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_path = self.data_ingestion_artifact.trained_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            train_df = self.read_data(train_path)
            test_df = self.read_data(test_path)

            # Validate schema
            schema_ok = all([
                self.validate_number_of_columns(train_df),
                self.validate_number_of_columns(test_df),
                self.is_numerical_column_exist(train_df),
                self.is_numerical_column_exist(test_df),
                self.is_categorical_column_exist(train_df),
                self.is_categorical_column_exist(test_df)
            ])

            if not schema_ok:
                logging.error("❌ Schema validation failed.")

            # Drift
            drift_ok = self.detect_dataset_drift(
                base_df=train_df.drop(columns=[self._schema_config["target_column"][0]]),
                current_df=test_df.drop(columns=[self._schema_config["target_column"][0]])
            )

            # Prior prob
            _ = self.detect_prior_probability_drift(train_df, test_df, self._schema_config["target_column"][0])

            # Concept drift
            concept_acc = self.detect_concept_drift(train_df.copy(), test_df.copy(), self._schema_config["target_column"])
            concept_ok = concept_acc >= 0.7

            if not concept_ok:
                logging.warning("⚠️ Concept drift detected. Accuracy = %.2f", concept_acc)

            validation_status = schema_ok and drift_ok and concept_ok

            # Copy validated files
            validated_dir = DATA_VALIDATION_VALIDATED_PATH
            os.makedirs(validated_dir, exist_ok=True)

            validated_train_path = os.path.join(validated_dir, "train.csv")
            validated_test_path = os.path.join(validated_dir, "test.csv")

            shutil.copy(train_path, validated_train_path)
            shutil.copy(test_path, validated_test_path)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=validated_train_path,
                valid_test_file_path=validated_test_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            write_yaml_file(file_path=OUTPUT_PATH, content=artifact.__dict__)
            logging.info(f"✅ Data Validation Artifact: \n\n===========>>>>>{artifact}<==============")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion_artifact = DataIngestionArtifact(
        trained_file_path=data_ingestion_config.training_file_path,
        test_file_path=data_ingestion_config.testing_file_path
    )

    data_validation_config = DataValidationConfig()

    validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=data_validation_config
    )
    validation.initiate_data_validation()







# import os
# import sys
# import pandas as pd
# from scipy.stats import ks_2samp
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score


# from insurance.constants.training_pipeline import SCHEMA_FILE_PATH, OUTPUT_PATH
# from insurance.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
# from insurance.entity.config_entity import DataIngestionConfig, DataValidationConfig
# from insurance.exception import CustomException
# from insurance.logger import logging
# from insurance.utils.main_utils import read_yaml_file, write_yaml_file


# class DataValidation:
#     def __init__(self,
#                  data_ingestion_artifact: DataIngestionArtifact,
#                  data_validation_config: DataValidationConfig):
#         try:
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.data_validation_config = data_validation_config
#             self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
#         except Exception as e:
#             raise CustomException(e, sys)

#     def drop_zero_std_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
#         return dataframe.loc[:, dataframe.std() > 0]

#     def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
#         try:
#             expected_columns = len(self._schema_config["columns"])
#             actual_columns = len(dataframe.columns)
#             logging.info(f"Expected columns: {expected_columns}, Found: {actual_columns}")
#             return expected_columns == actual_columns
#         except Exception as e:
#             raise CustomException(e, sys)

#     def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
#         try:
#             numerical_columns = self._schema_config["numerical_columns"]
#             missing_numerical = [col for col in numerical_columns if col not in dataframe.columns]
#             if missing_numerical:
#                 logging.warning(f"Missing numerical columns: {missing_numerical}")
#                 return False
#             return True
#         except Exception as e:
#             raise CustomException(e, sys)

#     def is_categorical_column_exist(self, dataframe: pd.DataFrame) -> bool:
#         try:
#             categorical_columns = self._schema_config["categorical_columns"]
#             missing_categorical = [col for col in categorical_columns if col not in dataframe.columns]
#             if missing_categorical:
#                 logging.warning(f"Missing categorical columns: {missing_categorical}")
#                 return False
#             return True
#         except Exception as e:
#             raise CustomException(e, sys)

#     @staticmethod
#     def read_data(file_path: str) -> pd.DataFrame:
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise CustomException(e, sys)

#     def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
#         try:
#             status = True
#             report = {}

#             for column in base_df.columns:
#                 d1 = base_df[column].dropna()
#                 d2 = current_df[column].dropna()

#                 if d1.dtype == "object" or d2.dtype == "object":
#                     d1 = d1.astype(str)
#                     d2 = d2.astype(str)
#                     label_encoder = LabelEncoder()
#                     combined = pd.concat([d1, d2])
#                     label_encoder.fit(combined)
#                     d1 = label_encoder.transform(d1)
#                     d2 = label_encoder.transform(d2)

#                 statistic, p_value = ks_2samp(d1, d2)
#                 p_value = float(p_value)
#                 drift_status = bool(p_value < threshold)

#                 if drift_status:
#                     status = False

#                 report[column] = {
#                     "p_value": p_value,
#                     "drift_status": drift_status
#                 }

#             drift_report_file_path = self.data_validation_config.drift_report_file_path
#             os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
#             write_yaml_file(file_path=drift_report_file_path, content=report)

#             return status
#         except Exception as e:
#             raise CustomException(e, sys)

#     def detect_prior_probability_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str) -> dict:
#         try:
#             base_dist = base_df[target_col].value_counts(normalize=True)
#             current_dist = current_df[target_col].value_counts(normalize=True)

#             all_classes = set(base_dist.index).union(current_dist.index)
#             drift_report = {}

#             for cls in all_classes:
#                 base_p = float(base_dist.get(cls, 0.0))
#                 current_p = float(current_dist.get(cls, 0.0))
#                 diff = abs(base_p - current_p)

#                 drift_report[str(cls)] = {
#                     "base_probability": base_p,
#                     "current_probability": current_p,
#                     "absolute_difference": diff
#                 }

#             prior_drift_path = self.data_validation_config.prior_drift_report_file_path
#             os.makedirs(os.path.dirname(prior_drift_path), exist_ok=True)
#             write_yaml_file(file_path=prior_drift_path, content=drift_report)

#             return drift_report
#         except Exception as e:
#             raise CustomException(e, sys)

#     def detect_concept_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str) -> float:
#         try:
#             if isinstance(target_col, (list, tuple)):
#                 target_col = target_col[0]

#             label_encoder = LabelEncoder()
#             base_df[target_col] = label_encoder.fit_transform(base_df[target_col])
#             current_df[target_col] = label_encoder.transform(current_df[target_col])

#             X_train = base_df.drop(columns=[target_col])
#             y_train = base_df[target_col]
#             X_test = current_df.drop(columns=[target_col])
#             y_test = current_df[target_col]

#             # Encode categorical features
#             for col in X_train.columns:
#                 if X_train[col].dtype == 'object' or X_test[col].dtype == 'object':
#                     enc = LabelEncoder()
#                     all_values = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
#                     enc.fit(all_values)
#                     X_train[col] = enc.transform(X_train[col].astype(str))
#                     X_test[col] = enc.transform(X_test[col].astype(str))

#             model = LogisticRegression(max_iter=10000)
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)

#             accuracy = accuracy_score(y_test, predictions)
#             return float(accuracy)
#         except Exception as e:
#             raise CustomException(e, sys)

#     def initiate_data_validation(self) -> DataValidationArtifact:
#         try:
#             error_message = ""

#             train_file_path = self.data_ingestion_artifact.trained_file_path
#             test_file_path = self.data_ingestion_artifact.test_file_path

#             train_df = self.read_data(train_file_path)
#             test_df = self.read_data(test_file_path)

#             # Basic schema checks
#             schema_checks = all([
#                 self.validate_number_of_columns(train_df),
#                 self.validate_number_of_columns(test_df),
#                 self.is_numerical_column_exist(train_df),
#                 self.is_numerical_column_exist(test_df),
#                 self.is_categorical_column_exist(train_df),
#                 self.is_categorical_column_exist(test_df),
#             ])

#             if not schema_checks:
#                 error_message += "❌ Schema validation failed.\n"

#             # Drift check
#             drift_passed = self.detect_dataset_drift(
#                 train_df.drop(columns=[self._schema_config["target_column"][0]]),
#                 test_df.drop(columns=[self._schema_config["target_column"][0]])
#             )

#             # Prior prob drift
#             prior_drift = self.detect_prior_probability_drift(
#                 train_df, test_df, self._schema_config["target_column"][0]
#             )

#             # Concept drift (accuracy score)
#             concept_accuracy = self.detect_concept_drift(
#                 train_df.copy(), test_df.copy(), self._schema_config["target_column"]
#             )
#             concept_passed = concept_accuracy >= 0.7

#             if not concept_passed:
#                 logging.warning("⚠️ Concept Drift Detected")

#             # Final validation status
#             validation_status = schema_checks and drift_passed and concept_passed
            


#             if not validation_status:
#                 logging.error(error_message + "Validation failed.")

#             data_validation_artifact = DataValidationArtifact(
#                 validation_status=validation_status,
#                 valid_train_file_path=train_file_path,
#                 valid_test_file_path=test_file_path,
#                 drift_report_file_path=self.data_validation_config.drift_report_file_path
#             )
            
#             write_yaml_file(
#                 file_path=OUTPUT_PATH,  # e.g., artifacts/validation.yaml
#                 content=data_validation_artifact.__dict__
#             )

#             logging.info(f"✅ Data Validation Artifact: {data_validation_artifact}")
#             return data_validation_artifact

#         except Exception as e:
#             raise CustomException(e, sys)


# if __name__ == "__main__":
#     data_ingestion_config = DataIngestionConfig()
#     data_ingestion_artifact = DataIngestionArtifact(
#         trained_file_path=data_ingestion_config.training_file_path,
#         test_file_path=data_ingestion_config.testing_file_path
#     )

#     data_validation_config = DataValidationConfig()
#     data_validation = DataValidation(
#         data_ingestion_artifact=data_ingestion_artifact,
#         data_validation_config=data_validation_config
#     )
#     data_validation.initiate_data_validation()
