import os
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve

from . import config
from .logger import ExperimentLogger
from .data_loader import load_datasets
from .scoring import anomaly_scores



def calculate_anomaly_threshold(model, val_ds):
    """
    Uses the validation set (normal data) to find the baseline error.
    Threshold = Mean + (2 * Standard Deviation)
    """
    print("Calculating anomaly threshold from validation set...")
    errors = []

    for inputs, targets in val_ds:
        reconstructions = model.predict(inputs, verbose=0)
        batch_errors = anomaly_scores(targets, reconstructions)
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



def run_latency_benchmark(model, test_ds):
    """
    Measures inference time to ensure real-time compatibility.
    """
    print("\nRunning Latency Benchmark...")
    
    # Optimize by calling the model directly
    @tf.function(reduce_retracing=True)
    def model_step(x):
        return model(x, training=False)

    # For warm up
    batch = next(iter(test_ds.take(1)))
    inputs = batch[0]
    _ = model_step(inputs)
    
    start_time = time.perf_counter()
    total_images = 0
    iterations = 5
    
    for i in range(iterations):
        for inputs, _, _ in test_ds.take(10):
            _ = model_step(inputs)
            total_images += len(inputs)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = total_images / total_time
    ms_per_image = (total_time / total_images) * 1000
    
    print(f"Latency: {ms_per_image:.2f}ms per image")
    print(f"Throughput: {fps:.2f} FPS (Target: >30 FPS)")
    return fps, ms_per_image



def evaluate():
    """
    Evaluates the Autoencoder and detects anomalies.
    """
    
    print("Loading datasets and the model...")
    _, val_ds, test_ds, _ = load_datasets()
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.keras")

    if not config.MODEL_ARTIFACT and not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first "
              f"or set config.MODEL_ARTIFACT.")
        return

    ExperimentLogger.init(
        job_type="evaluate",
        name=f"{config.RUN_NAME}-eval",
        tags=list(config.RUN_TAGS) + ["evaluate"],
    )

    # Link the model artifact as a run input and fetch its checkpoint.
    fetched_path = ExperimentLogger.use_model_artifact(config.MODEL_ARTIFACT, config.OUTPUT_DIR)
    if fetched_path:
        model_path = fetched_path

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return

    model = tf.keras.models.load_model(model_path, compile=False)
    threshold = calculate_anomaly_threshold(model, val_ds)
    
    print("Gathering performance metrics...")
    y_true = []
    y_scores = []
    
    for images, targets, paths in test_ds:
        for p in paths.numpy():
            path_str = p.decode("utf-8").lower()
            is_anomaly = 1 if "normal" not in path_str else 0
            y_true.append(is_anomaly)
        
        reconstructions = model.predict(images, verbose=0)
        mse_scores = anomaly_scores(targets, reconstructions)
        y_scores.extend(mse_scores)

    roc_auc = roc_auc_score(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    prevalence = float(np.mean(y_true))
    ap_lift = float(avg_precision - prevalence)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
    feasible = prec_curve >= 0.90
    recall_at_p90 = float(np.max(rec_curve[feasible])) if np.any(feasible) else 0.0

    y_pred = [1 if s > threshold else 0 for s in y_scores]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fps, latency_ms = run_latency_benchmark(model, test_ds)

    # Save metadata
    metadata_path = os.path.join(config.OUTPUT_DIR, "best_model_metadata.json")
    metadata = {
        "threshold": float(threshold),
        "score_method": config.SCORE_METHOD,
        "topk_percent": config.TOPK_PERCENT,
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "avg_precision": avg_precision,
        "prevalence": prevalence,
        "ap_lift": ap_lift,
        "recall_at_p90": recall_at_p90,
        "fps": fps,
        "latency_ms": latency_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_path}")

    print(f"\n--- Production Evaluation ---")
    print(f"ROC-AUC:         {roc_auc:.4f} (Target: 0.75-0.85)")
    print(f"Avg Precision:   {avg_precision:.4f} (Prevalence floor: {prevalence:.4f})")
    print(f"AP Lift:         {ap_lift:+.4f} (AP - prevalence; >0 beats chance)")
    print(f"Recall @ P90:    {recall_at_p90:.4f} (max recall while precision >= 0.90)")
    print(f"Precision:       {precision:.4f} (False Alarm Rate: {1-precision:.4f})")
    print(f"Recall (Safety): {recall:.4f} (Missed Cracks: {1-recall:.4f})")

    # Integrate wandb experiment logger if enabled (log metrics)
    ExperimentLogger.log_production_metrics({
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "prevalence": prevalence,
        "ap_lift": ap_lift,
        "recall_at_p90": recall_at_p90,
        "precision": precision,
        "recall": recall,
        "fps": fps,
        "latency_ms": latency_ms,
        "threshold": float(threshold),
        "y_true": y_true,
        "y_scores": y_scores
    })

    # Evaluate on test set
    print("Evaluating on mixed test set...")
    results = []

    heatmap_dir = os.path.join(config.OUTPUT_DIR, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # Process samples
    sample_count = 0
    for inputs, targets, paths in test_ds:
        reconstructions = model.predict(inputs, verbose=0)
        mse_scores = anomaly_scores(targets, reconstructions)

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

    # Integrate wandb experiment logger if enabled (Upload metadata + heatmaps)
    ExperimentLogger.upload_eval_artifacts(metadata_path, heatmap_dir)



if __name__ == "__main__":
    evaluate()