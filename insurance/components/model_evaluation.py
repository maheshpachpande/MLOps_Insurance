from insurance.exception import CustomException
from insurance.logger import logging

from insurance.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)

from insurance.entity.config_entity import (
    DataValidationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

import os, sys
import pandas as pd
from dataclasses import asdict

from insurance.ml.metric.classification_metrics import get_classification_score
from insurance.ml.model.estimator import InsuranceModel, ModelResolver, TargetValueMapping

from insurance.utils.main_utils import (
    save_object,
    load_object,
    read_yaml_file,
    write_yaml_file
)

from insurance.constants.training_pipeline import TARGET_COLUMN, DATA_VALIDATION_VALIDATED_PATH


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            print("🚀 Starting model evaluation")

            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path
            
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df, test_df])
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(), inplace=True)
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True

            # If no existing model to compare with
            if not model_resolver.is_model_exists():
                artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=0.0,
                    best_model_path="",
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None
                )
                os.makedirs(os.path.dirname(self.model_eval_config.report_file_path), exist_ok=True)
                write_yaml_file(
                    file_path=self.model_eval_config.report_file_path,
                    content=asdict(artifact)
                )
                print(f"📄 Model evaluation YAML written (no baseline model).")
                return artifact

            # Load models and predict
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric.f1_score - latest_metric.f1_score
            is_model_accepted = improved_accuracy > self.model_eval_config.change_threshold

            # Final artifact
            artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric
            )

            # Convert nested metrics to dict
            artifact_dict = asdict(artifact)
            artifact_dict["train_model_metric_artifact"] = asdict(trained_metric)
            if latest_metric:
                artifact_dict["best_model_metric_artifact"] = asdict(latest_metric)

            # Write to YAML
            os.makedirs(os.path.dirname(self.model_eval_config.report_file_path), exist_ok=True)
            write_yaml_file(
                file_path=self.model_eval_config.report_file_path,
                content=artifact_dict
            )

            print(f"✅ Model evaluation YAML saved at: {self.model_eval_config.report_file_path}")
            print(f"📊 Improved Accuracy: {improved_accuracy:.4f}")
            logging.info(f"Model Evaluation Artifact:\n{artifact_dict}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)


# ---------- 🔍 Entry Point for Testing Standalone Script ----------
if __name__ == "__main__":
    print("🟢 Model Evaluation Script Started")

    try:
        data_validation_config = DataValidationConfig()
        model_trainer_config = ModelTrainerConfig()
        model_eval_config = ModelEvaluationConfig()

        val = read_yaml_file("artifact/data_validation/validation.yaml")
        metric = read_yaml_file("artifact/model_trainer/model_trainer_artifact.yaml")

        data_validation_artifact = DataValidationArtifact(
            validation_status=val["validation_status"],
            valid_train_file_path=data_validation_config.valid_train_file_path,
            valid_test_file_path=data_validation_config.valid_test_file_path,
            drift_report_file_path=data_validation_config.drift_report_file_path
        )

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=model_trainer_config.trained_model_file_path,
            train_metric_artifact=metric["train_metric_artifact"],
            test_metric_artifact=metric["test_metric_artifact"]
        )

        model_eval = ModelEvaluation(
            model_eval_config=model_eval_config,
            data_validation_artifact=data_validation_artifact,
            model_trainer_artifact=model_trainer_artifact
        )

        result = model_eval.initiate_model_evaluation()
        print("🎯 Final ModelEvaluationArtifact:")
        print(result)

    except Exception as e:
        print("❌ Exception occurred during model evaluation")
        raise CustomException(e, sys)
