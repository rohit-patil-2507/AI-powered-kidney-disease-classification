import tensorflow as tf
import tf2onnx
import os

# Define paths
models_to_convert = [
    ("model/model.h5", "model/model.onnx"),
    ("model/ct_verification_model.h5", "model/ct_verification_model.onnx")
]

def convert_model():
    # Define the input signature. Based on your preprocessing, the input shape is (None, 224, 224, 3)
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

    for h5_path, onnx_path in models_to_convert:
        if not os.path.exists(h5_path):
            print(f"Warning: Model file not found at {h5_path}. Skipping.")
            continue

        print(f"Loading Keras model from {h5_path}...")
        model = tf.keras.models.load_model(h5_path, compile=False)

        print(f"Converting {h5_path} to ONNX... This may take a moment.")
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_path)
        
        print(f"Successfully converted! ONNX model saved to: {onnx_path}\n")

if __name__ == "__main__":
    convert_model()