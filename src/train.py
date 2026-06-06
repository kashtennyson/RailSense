import os
import numpy as np
import tensorflow as tf
from .model import get_model
from .logger import ExperimentLogger
from .data_loader import load_datasets
from . import config



def structural_loss(y_true, y_pred):
    """
    Hybrid loss: Alpha * (1 - SSIM) + (1 - Alpha) * L1
    Focuses on structural integrity over raw pixel values.
    """
    alpha = 0.8
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    return alpha * ssim_loss + (1 - alpha) * l1_loss



def train():
    # Integrate wandb experiment logger if enabled (initiate)
    ExperimentLogger.init()

    # Prepare data
    print("Loading datasets...")
    train_ds, val_ds, _ = load_datasets()

    # Build model
    print("Building model...")
    model = get_model()

    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    model.compile(
        optimizer=optimizer,
        loss=structural_loss,
        metrics=['mae']
    )

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.keras")

    # Define callbacks
    callbacks = [
        # Saves the model every time validation accuracy improves
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Stops training if validation accuracy doesn't improve for 10 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Lowers learning rate if progress stalls after 5 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Integrate wandb experiment logger if enabled
        ExperimentLogger.get_visual_callback(val_ds), # Visual logger
        *ExperimentLogger.get_train_callbacks() # Callbacks
    ]

    # Run training
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    # Integrate wandb experiment logger if enabled (upload model)
    ExperimentLogger.upload_model(model_path)

    return history



if __name__ == "__main__":
    train()