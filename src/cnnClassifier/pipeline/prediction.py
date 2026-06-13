from pathlib import Path
import os

import numpy as np
from PIL import Image


import base64
import io
import uuid
import hashlib

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    lime_image = None
    mark_boundaries = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class PredictionPipeline:
    """Prediction pipeline."""


    class_names = ("Cyst", "Normal", "Stone", "Tumor")
    _cached_main_model = None
    _cached_verification_model = None
    _cached_onnx_model = None
    _cached_verification_onnx_model = None
    _cached_grad_model = None
    _cached_lime_overlays = {}
    _cached_attention_overlays = {}
    _cached_gradcam_overlays = {}

    def __init__(self, filename, model_path: str = "model/model.onnx"):
        self.filename = filename
        self.model_path = self._resolve_model_path(model_path)
        self._model = None

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        target_onnx = Path("model/model.onnx")
        if target_onnx.exists():
            return target_onnx

        local_onnx = Path("artifacts/training/model.onnx")
        if local_onnx.exists():
            import shutil
            os.makedirs("model", exist_ok=True)
            shutil.copy(local_onnx, target_onnx)
            print(f"Copied local ONNX model from {local_onnx} to {target_onnx}")
            return target_onnx

        return target_onnx

    def _load_tensorflow(self):
        # TensorFlow import is done lazily.
        # On some Windows setups TF tries to load optional plugins (e.g. tfdml_plugin.dll).
        # If that fails, we raise a clear error so apps return a useful message.
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
        except Exception as e:
            raise RuntimeError(
                "TensorFlow import failed. This is commonly caused by missing Windows TF plugin DLLs "
                "(e.g., tensorflow-plugins/tfdml_plugin.dll).\n"
                f"Original error: {e}"
            )

        return tf, load_model

    def load_model(self):
        keras_model_path = self.model_path.with_suffix(".h5")
        if not keras_model_path.exists():
            local_h5 = Path("artifacts/training/model.h5")
            if local_h5.exists():
                import shutil
                os.makedirs("model", exist_ok=True)
                shutil.copy(local_h5, keras_model_path)
                print(f"Copied local Keras model from {local_h5} to {keras_model_path}")
            elif self.model_path.suffix == ".h5" and self.model_path.exists():
                keras_model_path = self.model_path

        if not keras_model_path.exists():
            raise FileNotFoundError(
                f"Keras model file not found at {keras_model_path}. Required for XAI visualizations (Grad-CAM/Attention)."
            )

        if PredictionPipeline._cached_main_model is None:
            _, load_model = self._load_tensorflow()
            try:
                PredictionPipeline._cached_main_model = load_model(str(keras_model_path), compile=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load the Keras model from '{keras_model_path}'. "
                    f"Ensure the file is a valid .h5 Keras model, not an ONNX file or a Git LFS pointer.\n\n"
                    f"Original Keras Error: {e}"
                ) from e

        self._model = PredictionPipeline._cached_main_model
        return self._model

    def load_onnx_model(self):
        onnx_path = self.model_path.with_suffix(".onnx")
        if not onnx_path.exists():
            return None

        if PredictionPipeline._cached_onnx_model is None:
            if ort is None:
                print("Warning: onnxruntime not installed. Falling back to TensorFlow.")
                return None
            # Providers: Use CPUExecutionProvider (or add CUDAExecutionProvider if GPU is available)
            PredictionPipeline._cached_onnx_model = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        return PredictionPipeline._cached_onnx_model

    def validate_image_quality(self, filename=None):
        """Validates if the image is a suitable candidate for prediction (e.g., not blurry/corrupted)."""
        image_path = filename or self.filename
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            raise ValueError("Instructions: Invalid or corrupted image format. Please upload a supported image type (e.g., JPEG, PNG).")

        try:
            with Image.open(image_path) as img:
                # Check if the image is a colorful photo instead of a grayscale scan
                img_rgb = img.convert('RGB')
                img_array_rgb = np.asarray(img_rgb, dtype=np.float32)
                r, g, b = img_array_rgb[:,:,0], img_array_rgb[:,:,1], img_array_rgb[:,:,2]
                color_diff = np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b))
                
                if color_diff > 10.0:
                    raise ValueError("Instructions: Uploaded image appears to be a colorful photo. Please upload a grayscale Kidney CT scan.")

                img_gray = img.convert('L')
                img_array = np.asarray(img_gray, dtype=np.float32)

                # Check if image is blank/solid color (low variance)
                std_dev = np.std(img_array)
                if std_dev < 10:
                    raise ValueError("Instructions: The uploaded image appears to be blank or a solid color. Please upload a valid, clear Kidney CT/MRI scan.")

                # Check for extreme blurriness using Laplacian variance
                try:
                    import cv2
                    img_cv = np.asarray(img_gray, dtype=np.uint8)
                    blur_metric = cv2.Laplacian(img_cv, cv2.CV_64F).var()
                except ImportError:
                    import scipy.ndimage
                    blur_metric = scipy.ndimage.laplace(img_array).var()

                if blur_metric < 50:
                    raise ValueError("Uploaded image is too blurry. Please ensure the kidney scan is in focus and clear.")
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise ValueError(f"Instructions: Failed to process image for validation. Ensure it is a valid Kidney scan. Error: {str(e)}")

    def verify_ct_scan(self, filename=None):
        """Verifies if the uploaded image is a valid CT scan using a secondary model."""
        image_path = filename or self.filename
        
        verification_onnx_path = Path("model/ct_verification_model.onnx")
        processed = self.preprocess_image(filename)

        if verification_onnx_path.exists() and ort is not None:
            if PredictionPipeline._cached_verification_onnx_model is None:
                PredictionPipeline._cached_verification_onnx_model = ort.InferenceSession(
                    str(verification_onnx_path), providers=['CPUExecutionProvider']
                )
            
            ort_session = PredictionPipeline._cached_verification_onnx_model
            input_name = ort_session.get_inputs()[0].name
            batch_data = processed["batched_image"].astype(np.float32)
            probabilities = ort_session.run(None, {input_name: batch_data})[0][0]
        else:
            if PredictionPipeline._cached_verification_model is None:
                verification_model_path = Path("model/ct_verification_model.h5")
                if not verification_model_path.exists():
                    print(f"Warning: CT verification model not found at {verification_model_path}. Skipping verification.")
                    return

                tf, load_model = self._load_tensorflow()
                try:
                    PredictionPipeline._cached_verification_model = load_model(str(verification_model_path), compile=False)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "signature not found" in error_msg or "synchronously open" in error_msg:
                        if verification_model_path.exists():
                            os.remove(verification_model_path)
                        print(f"CT verification model was corrupted and has been deleted. Restart the server to redownload {verification_model_path}.")
                    else:
                        print(f"Failed to load CT verification model: {e}")
                    return

            verification_model = PredictionPipeline._cached_verification_model
            probabilities = verification_model(processed["batched_image"], training=False)[0].numpy()
        
        # Assuming index 0 is non-CT and index 1 is CT, or single sigmoid output > 0.5 is CT
        confidence = float(probabilities[0] if len(probabilities) == 1 else probabilities[1])
        
        if confidence < 0.5:
            raise ValueError("Uploaded image is not a valid kidney CT scan.")

    def preprocess_image(self, filename=None):
        is_default = (filename is None or filename == self.filename)
        if is_default and hasattr(self, '_cached_preprocessed'):
            return self._cached_preprocessed
            
        image_path = filename or self.filename
        with Image.open(image_path) as img:
            resized_image = img.convert("RGB").resize((224, 224))
        image_array = np.asarray(resized_image, dtype=np.float32)
        normalized = image_array / 255.0
        batched = np.expand_dims(normalized, axis=0)

        result = {
            "resized_image": resized_image,
            "image_array": image_array,
            "normalized_image": normalized,
            "batched_image": batched,
        }
        
        if is_default:
            self._cached_preprocessed = result
            
        return result

    def predict_detailed(self, filename=None):
        processed = self.preprocess_image(filename)
        
        ort_session = self.load_onnx_model()
        if ort_session is not None:
            input_name = ort_session.get_inputs()[0].name
            batch_data = processed["batched_image"].astype(np.float32)
            probabilities = ort_session.run(None, {input_name: batch_data})[0][0]
        else:
            model = self.load_model()
            probabilities = model(processed["batched_image"], training=False)[0].numpy()
            
        probabilities = np.asarray(probabilities, dtype=float)

        if len(probabilities) == 1:
            prob_tumor = probabilities[0]
            prob_normal = 1.0 - prob_tumor
            predicted_index = 1 if prob_tumor > 0.5 else 0
            confidence = prob_tumor if predicted_index == 1 else prob_normal
            probs_dict = {
                "Normal": float(prob_normal),
                "Tumor": float(prob_tumor)
            }
            prediction = "Tumor" if predicted_index == 1 else "Normal"
        else:
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            probs_dict = {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(self.class_names[:len(probabilities)])
            }
            
            prediction = self.class_names[predicted_index]

        if confidence < 0.60:
            prediction = "Uncertain"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs_dict,
            "predicted_index": predicted_index,
        }

    def get_last_conv_layer_name(self):
        if hasattr(PredictionPipeline, '_cached_last_conv_layer_name'):
            return PredictionPipeline._cached_last_conv_layer_name
            
        tf, _ = self._load_tensorflow()
        model = self.load_model()

        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                PredictionPipeline._cached_last_conv_layer_name = layer.name
                return layer.name

        raise ValueError("Could not find a Conv2D layer for Grad-CAM.")

    @staticmethod
    def _to_data_url_png(pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def make_gradcam_heatmap(self, filename=None, class_index=None, layer_name=None):

        tf, _ = self._load_tensorflow()
        model = self.load_model()
        processed = self.preprocess_image(filename)
        # Explicitly cast to tensor to guarantee GradientTape tracking
        image_batch = tf.convert_to_tensor(processed["batched_image"], dtype=tf.float32)
        
        target_layer_name = layer_name or self.get_last_conv_layer_name()

        # Optimize by caching the grad_model to avoid rebuilding the graph every time
        if PredictionPipeline._cached_grad_model is None:
            last_layer = model.layers[-1]
            grad_output = model.output
            
            # Use pre-activation logits to prevent vanishing gradients on highly confident predictions
            if isinstance(last_layer, tf.keras.layers.Dense):
                # Reconstruct the Dense layer without activation to avoid tf_fn errors on KerasTensors
                linear_layer = tf.keras.layers.Dense(
                    units=last_layer.units,
                    use_bias=last_layer.use_bias,
                    activation=None,
                    name=f"{last_layer.name}_linear_{uuid.uuid4().hex[:8]}"
                )
                grad_output = linear_layer(last_layer.input)
                linear_layer.set_weights(last_layer.get_weights())
            elif isinstance(last_layer, tf.keras.layers.Activation) or type(last_layer).__name__ == 'Softmax':
                grad_output = last_layer.input
                
            PredictionPipeline._cached_grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(target_layer_name).output, grad_output],
            )
            
        grad_model = PredictionPipeline._cached_grad_model

        with tf.GradientTape() as tape:
            # Forward pass is cleanly executed and tracked
            outputs = grad_model(image_batch, training=False)
            
            if isinstance(outputs, dict):
                outputs = list(outputs.values())
                
            conv_outputs = outputs[0]
            predictions = outputs[1]
            
            if isinstance(predictions, list):
                predictions = predictions[0]
            if isinstance(conv_outputs, list):
                conv_outputs = conv_outputs[0]

            selected_index = class_index
            if selected_index is None:
                if predictions.shape[-1] == 1:
                    selected_index = 0
                else:
                    selected_index = tf.argmax(predictions[0])
            class_channel = predictions[:, selected_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to keep only positive influences and normalize safely
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    def make_gradcam_overlay_base64(
        self,
        filename=None,
        class_index=None,
        layer_name=None,
        alpha: float = 0.38,
        cmap: str = "jet",
    ) -> str:
        """Return a base64 PNG data-URL of Grad-CAM overlay for the given image."""
        processed = self.preprocess_image(filename)
        img_arr = processed["normalized_image"]
        
        img_hash = hashlib.md5(img_arr.tobytes()).hexdigest()
        cache_key = f"{img_hash}_{class_index}_{layer_name}_{alpha}_{cmap}"
        
        if cache_key in PredictionPipeline._cached_gradcam_overlays:
            return PredictionPipeline._cached_gradcam_overlays[cache_key]
            
        tf, _ = self._load_tensorflow()
        _ = tf  # keep TF lazy/import consistent

        # Get raw heatmap (0..1)
        heatmap = self.make_gradcam_heatmap(filename=filename, class_index=class_index, layer_name=layer_name)

        # Colorize heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        # Resize heatmap to match image size (224, 224)
        if heatmap_uint8.shape[:2] != (224, 224):
            if hasattr(Image, "Resampling"):
                resample_mode = Image.Resampling.BILINEAR
            else:
                resample_mode = Image.BILINEAR
            heatmap_img = Image.fromarray(heatmap_uint8).resize((224, 224), resample=resample_mode)
            heatmap_uint8 = np.array(heatmap_img)

        try:
            # Use highly optimized OpenCV for instantaneous color mapping instead of Matplotlib
            import cv2
            colormap = getattr(cv2, f"COLORMAP_{cmap.upper()}", cv2.COLORMAP_JET)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            colored = heatmap_rgb.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Colormap error: {e}")
            # Fallback to Matplotlib if OpenCV fails
            try:
                import matplotlib as mpl
                import matplotlib.cm as cm
                if hasattr(mpl, 'colormaps'):
                    cmap_fn = mpl.colormaps[cmap]
                else:
                    cmap_fn = cm.get_cmap(cmap)
                rgba = cmap_fn(heatmap_uint8)
                colored = rgba[..., :3].astype(np.float32)
            except Exception:
                # Fallback: grayscale -> red channel only
                colored = np.zeros((heatmap_uint8.shape[0], heatmap_uint8.shape[1], 3), dtype=np.float32)
                colored[..., 0] = heatmap_uint8.astype(np.float32) / 255.0

        overlay = (1 - alpha) * img_arr + alpha * colored
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

        result_b64 = self._to_data_url_png(overlay_img)
        PredictionPipeline._cached_gradcam_overlays[cache_key] = result_b64
        return result_b64

    def make_preprocess_previews_base64(self, filename=None) -> dict:
        """Return model input previews as base64 PNG data-URLs."""
        processed = self.preprocess_image(filename)
        resized = processed["resized_image"]

        # normalized preview: show as image where [0..1] -> [0..255]
        normalized = processed["normalized_image"]
        normalized_img = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        normalized_pil = Image.fromarray(normalized_img)

        return {
            "resized": self._to_data_url_png(resized),
            "normalized_preview": self._to_data_url_png(normalized_pil),
        }

    def make_attention_overlay_base64(self, filename=None, alpha: float = 0.5, cmap: str = "magma") -> str:
        """
        Generates a Saliency/Attention Map visualization (gradients of the output w.r.t the input image)
        and returns it as a base64 encoded data URI.
        """
        processed = self.preprocess_image(filename)
        img_array = processed["normalized_image"]
        image_batch = processed["batched_image"]
        
        # Calculate hash for caching
        img_hash = hashlib.md5(image_batch.tobytes()).hexdigest()
        if not hasattr(PredictionPipeline, '_cached_attention_overlays'):
            PredictionPipeline._cached_attention_overlays = {}
        if img_hash in PredictionPipeline._cached_attention_overlays:
            return PredictionPipeline._cached_attention_overlays[img_hash]

        tf, _ = self._load_tensorflow()
        model = self.load_model()
        
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = model(image_tensor, training=False)
            
            if predictions.shape[-1] == 1:
                class_channel = predictions[0]
            else:
                pred_idx = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_idx]
                
        grads = tape.gradient(class_channel, image_tensor)
        
        # Calculate Saliency (Attention)
        attention = tf.reduce_max(tf.abs(grads), axis=-1)[0]
        
        # Normalize safely
        max_val = tf.reduce_max(attention)
        if max_val > 0:
            attention = attention / max_val
            
        attention_np = attention.numpy()
        attention_uint8 = np.uint8(255 * attention_np)
        
        try:
            import cv2
            colormap = getattr(cv2, f"COLORMAP_{cmap.upper()}", cv2.COLORMAP_MAGMA)
            attention_colored = cv2.applyColorMap(attention_uint8, colormap)
            attention_rgb = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
            colored = attention_rgb.astype(np.float32) / 255.0
        except Exception:
            colored = np.zeros((attention_uint8.shape[0], attention_uint8.shape[1], 3), dtype=np.float32)
            colored[..., 0] = attention_uint8.astype(np.float32) / 255.0
                
        overlay = (1 - alpha) * img_array + alpha * colored
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        
        result_b64 = self._to_data_url_png(overlay_img)
        PredictionPipeline._cached_attention_overlays[img_hash] = result_b64
        return result_b64

    def make_lime_overlay_base64(self, filename=None) -> str:
        """
        Generates a LIME (Local Interpretable Model-agnostic Explanations) visualization 
        and returns it as a base64 encoded data URI.
        """
        if lime_image is None or mark_boundaries is None:
            raise ImportError("LIME dependencies missing. Please run: pip install lime scikit-image")

        processed = self.preprocess_image(filename)
        img_array = processed["normalized_image"]
        image_batch = processed["batched_image"]
        
        img_hash = hashlib.md5(image_batch.tobytes()).hexdigest()
        if not hasattr(PredictionPipeline, '_cached_lime_overlays'):
            PredictionPipeline._cached_lime_overlays = {}
        if img_hash in PredictionPipeline._cached_lime_overlays:
            return PredictionPipeline._cached_lime_overlays[img_hash]

        explainer = lime_image.LimeImageExplainer()

        ort_session = self.load_onnx_model()
        model = self.load_model() if ort_session is None else None

        def predict_wrapper(images_batch):
            images_batch = images_batch.astype(np.float32)
            if ort_session is not None:
                input_name = ort_session.get_inputs()[0].name
                return ort_session.run(None, {input_name: images_batch})[0]
            else:
                predictions = model(images_batch, training=False)
                return predictions.numpy()

        explanation = explainer.explain_instance(
            img_array.astype('double'), 
            predict_wrapper, 
            top_labels=1, 
            hide_color=0, 
            num_samples=150,
            batch_size=50
        )

        temp_img, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        img_boundry = mark_boundaries(temp_img, mask)

        buf = io.BytesIO()
        import matplotlib.pyplot as plt
        plt.imsave(buf, img_boundry, format='png') 
        buf.seek(0)
        
        b64_encoded = base64.b64encode(buf.read()).decode('utf-8')
        result_b64 = f"data:image/png;base64,{b64_encoded}"
        
        PredictionPipeline._cached_lime_overlays[img_hash] = result_b64
        return result_b64
