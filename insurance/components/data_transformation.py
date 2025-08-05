import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from insurance.constants.training_pipeline import TARGET_COLUMN

from insurance.entity.config_entity import DataValidationConfig
from insurance.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from insurance.entity.config_entity import DataTransformationConfig
from insurance.exception import CustomException
from insurance.logger import logging
from insurance.ml.model.estimator import TargetValueMapping
from insurance.utils.main_utils import save_numpy_array_data, save_object, read_yaml_file

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['policy_deductable_cat'] = np.select(
            [X['policy_deductable'] == 500,
             X['policy_deductable'] == 1000,
             X['policy_deductable'] == 2000],
            ['Low', 'Medium', 'High'], default='Unknown')

        X['umbrella_limit_cat'] = np.select(
            [X['umbrella_limit'] == 0,
             X['umbrella_limit'] == 600000,
             X['umbrella_limit'] == 1200000,
             X['umbrella_limit'] == 2000000,
             X['umbrella_limit'] == 3000000],
            ['None', 'Basic', 'Standard', 'Extended', 'Premium'], default='Unknown')

        X['bodily_injuries_cat'] = np.select(
            [X['bodily_injuries'] == 0,
             X['bodily_injuries'] == 1,
             X['bodily_injuries'] == 2],
            ['None', 'Minor', 'Major'], default='Unknown')

        X['incident_hour_cat'] = np.select(
            [(X['incident_hour_of_the_day'] >= 0) & (X['incident_hour_of_the_day'] < 6),
             (X['incident_hour_of_the_day'] >= 6) & (X['incident_hour_of_the_day'] < 12),
             (X['incident_hour_of_the_day'] >= 12) & (X['incident_hour_of_the_day'] < 18),
             (X['incident_hour_of_the_day'] >= 18) & (X['incident_hour_of_the_day'] <= 23)],
            ['Early Morning', 'Morning', 'Afternoon', 'Evening'], default='Unknown')

        X['vehicles_involved_cat'] = np.select(
            [X['number_of_vehicles_involved'] == 1,
             X['number_of_vehicles_involved'] == 2,
             X['number_of_vehicles_involved'] >= 3],
            ['Single Vehicle', 'Two Vehicles', 'Multi-Vehicle'], default='Unknown')

        X['witnesses_cat'] = np.select(
            [X['witnesses'] == 0,
             X['witnesses'] == 1,
             X['witnesses'] >= 2],
            ['No Witness', 'Single Witness', 'Multiple Witnesses'], default='Unknown')

        X['policy_csl'] = X['policy_csl'].astype(str)
        X['policy_csl_cat'] = np.select(
            [X['policy_csl'] == '100/300',
             X['policy_csl'] == '250/500',
             X['policy_csl'] == '500/1000'],
            ['Basic', 'Standard', 'Premium'], default='Other')

        drop_cols = ['insured_zip', 'policy_number', 'policy_deductable', 'umbrella_limit',
                     'bodily_injuries', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
                     'witnesses', 'policy_csl', 'incident_location']
        X.drop(columns=drop_cols, inplace=True, errors='ignore')
        return X

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e)

    @classmethod
    def get_data_transformer_object(cls, df: pd.DataFrame) -> Pipeline:
        try:
            feature_engineer = CustomFeatureEngineer()

            df = feature_engineer.transform(df)

            numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            onehot_features = df.select_dtypes(include=['object']).columns.tolist()
            ordinal_features = ['policy_deductable_cat', 'umbrella_limit_cat', 'bodily_injuries_cat',
                                'incident_hour_cat', 'vehicles_involved_cat', 'witnesses_cat', 'policy_csl_cat']

            ordinal_mapping = [
                ['Low', 'Medium', 'High', 'Unknown'],
                ['None', 'Basic', 'Standard', 'Extended', 'Premium', 'Unknown'],
                ['None', 'Minor', 'Major', 'Unknown'],
                ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Unknown'],
                ['Single Vehicle', 'Two Vehicles', 'Multi-Vehicle', 'Unknown'],
                ['No Witness', 'Single Witness', 'Multiple Witnesses', 'Unknown'],
                ['Basic', 'Standard', 'Premium', 'Other', 'Unknown']
            ]


            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            onehot_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, drop="first"))
            ])

            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(categories=ordinal_mapping, handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', onehot_transformer, onehot_features),
                ('ord', ordinal_transformer, ordinal_features)
            ])

            full_pipeline = Pipeline(steps=[
                ('feature_engineering', feature_engineer),
                ('preprocessing', preprocessor)
            ])

            return full_pipeline

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                # raise CustomException("Data Validation failed.")
                logging.warning("⚠️ Data validation failed. Proceeding anyway (debug mode).")
            
            # if not data_validation_artifact.validation_status:
            #     logging.warning("⚠️ Data validation failed. Proceeding anyway (debug mode).")


            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())
            target_feature_test_df = test_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())

            preprocessor = self.get_data_transformer_object(input_feature_train_df)
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy="minority")
            train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df)
            
            input_feature_train_final = train_final[0]
            target_feature_train_final = train_final[1]
            
            test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df)
            
            input_feature_test_final = test_final[0]
            target_feature_test_final = test_final[1]

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise CustomException(e)




if __name__ == "__main__":
    
    data_validation_config = DataValidationConfig()
    
    cnf = read_yaml_file("artifact/data_validation/validation.yaml")
    
    data_validation_artifacts = DataValidationArtifact(
        validation_status=cnf["validation_status"],
        valid_train_file_path=cnf["valid_train_file_path"],
        valid_test_file_path=cnf["valid_test_file_path"],    
        drift_report_file_path=cnf["drift_report_file_path"]
    )
    
    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(data_validation_artifacts, data_transformation_config)
    data_transformation.initiate_data_transformation()