import os
import sys
import argparse
from src import config
from src.train import train
from src.evaluate import evaluate
from src.logger import ExperimentLogger
from src.inference import run_inference

# Supress info and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def set_global_seeds(seed):
    """
    Seeds Python, NumPy, and TensorFlow for reproducible runs.
    Note: GPU ops may still introduce minor non-determinism.
    """
    import random
    import numpy as np
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def main():
    """
    Main entry point for the Railway Track Crack Detection System.
    Handles CLI arguments and applies hyperparameter overrides to the central config.
    """

    parser = argparse.ArgumentParser(
        description="Railway Track Crack Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    # Primary action
    parser.add_argument(
        "action", 
        choices=["train", "evaluate", "predict", "both"],
        help="Actions to perform: 'train' from scratch, 'evaluate' the best saved model, 'predict' on an image, or 'both' (train and evaluate)."
    )

    # Hyperparameter overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Override number of training epochs"
        )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help="Override learning rate"
        )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Override batch size"
        )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for prediction"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=config.RUN_NAME,
        help="W&B run name (e.g. 'baseline-v0')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Global random seed for reproducibility"
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=config.USE_WANDB,
        help="Enable/disable Weights & Biases logging (use --no-wandb to disable)"
    )

    args = parser.parse_args()

    # Apply overrides if provided
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.BATCH_SIZE = args.batch_size
    config.RUN_NAME = args.run_name
    config.SEED = args.seed
    config.USE_WANDB = args.wandb

    # Seed everything for reproducibility
    set_global_seeds(config.SEED)

    # Print a summary
    print(f"\n--- Pipeline Configuration ---")
    print(f"Action:      {args.action.upper()}")
    print(f"Run Name:    {config.RUN_NAME}")
    print(f"Seed:        {config.SEED}")
    print(f"Epochs:      {config.EPOCHS}")
    print(f"LR:          {config.LEARNING_RATE}")
    print(f"Batch Size:  {config.BATCH_SIZE}")
    print(f"W&B Logging: {'Enabled' if config.USE_WANDB else 'Disabled'}")
    print(f"------------------------------\n")

    try:
        if args.action == "train":
            train()
        
        elif args.action == "evaluate":
            evaluate()
        
        elif args.action == "predict":
            if not args.image:
                print("Error: Please provide an image path using --image")
                sys.exit(1)
            run_inference(args.image)
        
        elif args.action == "both":
            print("\n--- Starting Full Pipeline ---")
            train()
            print("\n--- Training Complete. Starting Evaluation ---")
            evaluate()
        
        print("\n--- Pipeline execution finished ---")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n--- Pipeline stopped by user ---")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n--- Pipeline failed with the following message: {e}")
        sys.exit(1)
    
    finally:
        ExperimentLogger.finish()



if __name__ == "__main__":
    main()