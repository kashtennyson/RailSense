import os
import sys
import argparse
from src import config
from src.train import train
from src.evaluate import evaluate
from src.logger import ExperimentLogger

# Supress info and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        choices=["train", "evaluate", "all"],
        help="Action to perform: 'train' from scratch, 'evaluate' the best saved model, or 'all' (both)."
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

    args = parser.parse_args()

    # Apply overrides
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    # Print a summary
    print(f"\n--- Pipeline Configuration ---")
    print(f"Action:      {args.action.upper()}")
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
        elif args.action == "all":
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