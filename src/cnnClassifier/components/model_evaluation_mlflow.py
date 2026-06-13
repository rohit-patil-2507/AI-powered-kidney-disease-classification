import tensorflow as tf
from pathlib import Path
import os
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier import logger




class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config.mlflow_uri)
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = "Kidney-Disease-Classification"
        
        # Check if experiment exists and is not deleted
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None and exp.lifecycle_stage == "deleted":
            # Restore the deleted experiment using MlflowClient
            client = MlflowClient()
            client.restore_experiment(exp.experiment_id)
        elif exp is None:
            # Create new experiment if it doesn't exist
            mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            log_model = os.getenv(
                "MLFLOW_LOG_MODEL",
                "true" if tracking_url_type_store == "file" else "false"
            ).lower()

            if log_model in ("0", "false", "no"):
                logger.info(
                    "Skipping MLflow model artifact logging. Set MLFLOW_LOG_MODEL=true to upload/register the model."
                )
                return

            try:
                # Register model only if using remote tracking server
                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(
                        self.model,
                        "model",
                        registered_model_name="VGG16Model"
                    )
                else:
                    mlflow.keras.log_model(self.model, "model")
            except Exception as e:
                logger.warning(f"MLflow params and metrics were logged, but model artifact upload failed: {e}")
