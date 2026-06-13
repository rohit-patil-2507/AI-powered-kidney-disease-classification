from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "ONNX Model Conversion"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   import tensorflow as tf
   import tf2onnx
   import shutil
   from pathlib import Path
   
   model_path = Path("artifacts/training/model.h5")
   onnx_path = Path("artifacts/training/model.onnx")
   
   if model_path.exists():
       model = tf.keras.models.load_model(model_path, compile=False)
       spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
       tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(onnx_path))
       logger.info(f"Successfully converted model to ONNX: {onnx_path}")
       
       # Copy to deployment directory
       deploy_onnx_path = Path("model/model.onnx")
       deploy_onnx_path.parent.mkdir(parents=True, exist_ok=True)
       shutil.copy(onnx_path, deploy_onnx_path)
       logger.info(f"Copied ONNX model to deployment directory: {deploy_onnx_path}")
   else:
       logger.warning(f"Model not found at {model_path}, skipping ONNX conversion.")
       
   # Convert Secondary Verification Model
   verification_model_path = Path("model/ct_verification_model.h5")
   verification_onnx_path = Path("model/ct_verification_model.onnx")
   
   if verification_model_path.exists() and not verification_onnx_path.exists():
       logger.info(f"Converting secondary model {verification_model_path} to ONNX...")
       v_model = tf.keras.models.load_model(verification_model_path, compile=False)
       v_spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
       tf2onnx.convert.from_keras(v_model, input_signature=v_spec, opset=13, output_path=str(verification_onnx_path))
       logger.info(f"Successfully converted verification model to ONNX: {verification_onnx_path}")
       
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e
