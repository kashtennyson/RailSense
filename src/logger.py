import subprocess
import numpy as np
import tensorflow as tf
from . import config



if config.USE_WANDB:
    try:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger
    except ImportError:
        print("Warning: USE_WANDB is True but 'wandb' is not installed. Disabling logging.")
        config.USE_WANDB = False



def _get_git_commit():
    """Returns the current git commit hash, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"



class ExperimentLogger:
    class VisualProgress(tf.keras.callbacks.Callback):
        def __init__(self, val_ds, log_freq=5):
            super().__init__()
            self.val_ds = val_ds
            self.log_freq = log_freq



        def on_epoch_end(self, epoch, logs=None, **kwargs):
            if not config.USE_WANDB or not wandb.run: 
                return
            if (epoch + 1) % self.log_freq == 0:
                images, _ = next(iter(self.val_ds.take(1)))
                reconstructions = self.model.predict(images, verbose=0)
                
                samples = [
                    wandb.Image(
                        (np.hstack([images[i].numpy(),
                        reconstructions[i]]) * 255).astype(np.uint8), 
                        caption=f"Recon Ep{epoch+1}"
                        )
                    for i in range(min(4, len(images)))
                ]
                wandb.log({"visual_progress": samples}, commit=False)



    @staticmethod
    def init():
        """Starts a new experiment run if enabled."""
        if not config.USE_WANDB:
            print("Running in OFFLINE mode (W&B disabled).")
            return None
        
        return wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=config.RUN_NAME,
            job_type=config.RUN_JOB_TYPE,
            tags=config.RUN_TAGS,
            config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "weight_decay": config.WEIGHT_DECAY,
            "architecture": "ResNet Autoencoder",
            "loss_type": "SSIM + L1",
            "latent_dim": 512,
            "image_shape": config.IMAGE_SHAPE,
            "seed": config.SEED,
            "git_commit": _get_git_commit(),
            "dataset": "data"
            }
        )



    @staticmethod
    def get_train_callbacks():
        """Returns W&B callbacks ONLY if enabled, otherwise returns an empty list."""
        if not config.USE_WANDB:
            return []
        
        return [WandbMetricsLogger(log_freq=5)]



    @staticmethod
    def upload_model(file_path):
        """Uploads the final best model to W&B as an artifact if enabled."""
        if not config.USE_WANDB or not wandb.run:
            return

        print(f"Uploading final best model to W&B: {file_path}")
        # Create and upload a W&B artifact
        artifact = wandb.Artifact(
            name=f"model_{wandb.run.id}", 
            type="model",
            description="Best model saved during training"
        )
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)



    @staticmethod
    def log_production_metrics(metrics_dict):
        if not config.USE_WANDB or not wandb.run:
            return

        # Scalars-only dict (safe to log directly)
        scalar_metrics = {k: v for k, v in metrics_dict.items()
                        if k not in ("y_true", "y_scores")}
        wandb.log(scalar_metrics)

        if "y_true" in metrics_dict and "y_scores" in metrics_dict:
            y_true   = np.array(metrics_dict["y_true"],   dtype=int)
            y_scores = np.array(metrics_dict["y_scores"], dtype=float)

            # Reshape into (n_samples, n_classes) as wandb plot helpers expect
            y_probs = np.column_stack([1 - y_scores, y_scores])

            wandb.log({
                "roc_curve": wandb.plot.roc_curve(y_true, y_probs, labels=["normal", "anomaly"]),
                "pr_curve":  wandb.plot.pr_curve(y_true,  y_probs, labels=["normal", "anomaly"])
            })



    @staticmethod
    def get_visual_callback(val_ds, log_freq=5):
        """
        Creates and returns an instance of VisualProgress class.
        """
        return ExperimentLogger.VisualProgress(val_ds, log_freq)



    @staticmethod
    def finish():
        """Closes the run if enabled."""
        if config.USE_WANDB and wandb.run:
            wandb.finish()