import os
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score

from . import config
from .logger import ExperimentLogger
from .data_loader import load_datasets



def calculate_anomaly_threshold(model, val_ds):
    """
    Uses the validation set (normal data) to find the baseline error.
    Threshold = Mean + (2 * Standard Deviation)
    """
    print("Calculating anomaly threshold from validation set...")
    errors = []

    for inputs, targets in val_ds:
        reconstructions = model.predict(inputs, verbose=0)
        batch_errors = np.mean(
                            np.square(targets - reconstructions),
                            axis=(1, 2, 3)
                            )
        errors.extend(batch_errors)

    mean_err = np.mean(errors)
    std_err = np.std(errors)
    threshold = mean_err + (2 * std_err)
    
    print(f"Threshold Set: {threshold:.6f} (Mean: {mean_err:.6f}, Std: {std_err:.6f})")
    return threshold



def generate_heatmap(image, reconstruction):
    """
    Highlights areas where the reconstruction failed.
    """
    diff = np.abs(image - reconstruction)
    heatmap = np.mean(diff, axis=-1)
    return heatmap



def evaluate():
    """
    Evaluates the Autoencoder and detects anomalies.
    """
    
    print("Loading datasets and the model...")
    _, val_ds, test_ds = load_datasets()
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return

    model = tf.keras.models.load_model(model_path, compile=False)
    threshold = calculate_anomaly_threshold(model, val_ds)
    
    print("Gathering performance metrics...")
    y_true = []
    
    # Evaluate on test set
    print("Evaluating on mixed test set...")
    results = []

    heatmap_dir = os.path.join(config.OUTPUT_DIR, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # Process samples
    sample_count = 0
    for inputs, targets, paths in test_ds:
        reconstructions = model.predict(inputs, verbose=0)
        mse_scores = np.mean(
                        np.square(targets - reconstructions),
                        axis=(1, 2, 3)
                        )
        
        for i in range(len(inputs)):
            score = mse_scores[i]
            is_anomaly = score > threshold
            results.append((is_anomaly, score))
            
            # Save visual evidence for the first 10 anomalies found
            if is_anomaly and sample_count < 10:
                heatmap = generate_heatmap(targets[i].numpy(), reconstructions[i])
                
                # Plot: Original | Reconstruction | Heatmap
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(targets[i])
                axes[0].set_title("Original")
                axes[1].imshow(reconstructions[i])
                axes[1].set_title("Reconstruction")
                axes[2].imshow(heatmap, cmap='jet')
                axes[2].set_title(f"Heatmap (Score: {score:.4f})")
                
                for ax in axes: ax.axis('off')
                
                plt.savefig(os.path.join(heatmap_dir, f"anomaly_{sample_count}.png"))
                plt.close()
                sample_count += 1

    anomalies_detected = sum([1 for r in results if r[0]])
    print(f"\n--- Evaluation Summary ---")
    print(f"Total Test Images: {len(results)}")
    print(f"Anomalies Flagged: {anomalies_detected}")
    print(f"Detection Rate:    {(anomalies_detected/len(results))*100:.2f}%")
    print(f"Visual evidence saved to: {heatmap_dir}")



if __name__ == "__main__":
    evaluate()