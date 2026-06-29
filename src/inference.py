import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from . import config
from .scoring import compute_error_maps, score_from_error_maps



class AnomalyPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(config.OUTPUT_DIR, "best_model.keras")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train it first!")
            
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.threshold = 0.005
        self.score_method = config.SCORE_METHOD
        self.topk_percent = config.TOPK_PERCENT

        # Load threshold from metadata if available
        metadata_path = os.path.join(config.OUTPUT_DIR, "best_model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.threshold = metadata.get("threshold", self.threshold)
                    self.score_method = metadata.get("score_method", self.score_method)
                    self.topk_percent = metadata.get("topk_percent", self.topk_percent)
                print(f"Loaded threshold {self.threshold:.6f} (method '{self.score_method}') from {metadata_path}")
            except Exception as e:
                print(f"Warning: Could not load metadata, using default threshold. Error: {e}")
        else:
            print(f"No metadata found at {metadata_path}, using default threshold: {self.threshold}")

    def preprocess(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config.IMAGE_SHAPE)
        img = tf.cast(img, tf.float32) / 255.0
        return tf.expand_dims(img, axis=0)

    def predict(self, image_path, save_heatmap=True):
        input_tensor = self.preprocess(image_path)
        reconstruction = self.model.predict(input_tensor, verbose=0)

        error_map = compute_error_maps(input_tensor[0], reconstruction[0])
        mse_score = float(score_from_error_maps(
            error_map,
            method=self.score_method,
            topk_percent=self.topk_percent
        ))
        is_anomaly = mse_score > self.threshold
        confidence = 1.0 / (1.0 + np.exp(-(mse_score - self.threshold) / (self.threshold * 0.1)))

        result = {
            "status": "ANOMALY" if is_anomaly else "NORMAL",
            "score": float(mse_score),
            "confidence": float(confidence),
            "image_path": image_path,
            "heatmap_path": None
        }

        if save_heatmap:
            heatmap = error_map
            output_path = os.path.join(config.OUTPUT_DIR, "latest_prediction.png")
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(input_tensor[0])
            plt.title("Input")
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap, cmap='jet')
            plt.title(f"Heatmap (Score: {mse_score:.5f})")
            plt.savefig(output_path)
            plt.close()
            result["heatmap_path"] = output_path

        return result



def display_results(result):
    """
    Handles the terminal formatting for prediction results.
    """
    # Define colors for the status
    status = result['status']
    color_start = "\033[91m" if status == "ANOMALY" else "\033[92m"
    color_end = "\033[0m"

    print(f"\n--- {color_start}Inference Results{color_end} ---")
    print(f"File:       {os.path.basename(result['image_path'])}")
    print(f"Status:     {color_start}{status}{color_end}")
    print(f"Score:      {result['score']:.6f}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if result['heatmap_path']:
        print(f"Heatmap:    {os.path.abspath(result['heatmap_path'])}")
    
    print(f"{'-' * 25}\n")    



def run_inference(image_path):
    """
    A single-call function wrapper for main.py.
    """
    predictor = AnomalyPredictor() 
    result = predictor.predict(image_path)
    display_results(result)