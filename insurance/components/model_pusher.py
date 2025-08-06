
from insurance.exception import CustomException
from insurance.logger import logging
from insurance.entity.artifact_entity import (ModelPusherArtifact,
                                              ModelTrainerArtifact,
                                              ModelEvaluationArtifact)

from insurance.entity.config_entity import (ModelEvaluationConfig,
                                            ModelPusherConfig)
import os, sys
from insurance.ml.metric.classification_metrics import get_classification_score
from insurance.utils.main_utils import (save_object,
                                        load_object,
                                        write_yaml_file,
                                        read_yaml_file)

import shutil

class ModelPusher:

    def __init__(self,
                model_pusher_config:ModelPusherConfig,
                model_eval_artifact:ModelEvaluationArtifact):

        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise CustomException(e, sys)
    

    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            trained_model_path = self.model_eval_artifact.trained_model_path
            
            #Creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            #saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            #prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
            print(saved_model_path)
            print(model_file_path)
            
            return model_pusher_artifact
        except  Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        model_eval = read_yaml_file(file_path="artifact/model_evaluation/model_evaluation_artifact.yaml")
        
        model_evaluation = ModelEvaluationConfig()
        model_evaluation_artifact = ModelEvaluationArtifact(
            is_model_accepted=model_eval['is_model_accepted'],
            improved_accuracy=model_eval['improved_accuracy'],
            best_model_path=model_eval['best_model_path'],
            trained_model_path=model_eval['trained_model_path'],
            train_model_metric_artifact=model_eval['train_model_metric_artifact'],
            best_model_metric_artifact=model_eval['best_model_metric_artifact']
        )
        
        model_pusher = ModelPusher(model_eval_artifact=model_evaluation_artifact, model_pusher_config=ModelPusherConfig())
        model_pusher.initiate_model_pusher()
    except Exception as e:
        raise CustomException(e, sys)