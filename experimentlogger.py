import os
import json
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt

class ExperimentLogger:
    """Enhanced logging and metrics tracking for ML experiments"""
    
    def __init__(self, experiment_name: str = None):
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join("logs", experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging to file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()
        
        # Initialize metrics storage
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.start_time = None
    
    def log_experiment_config(self, config: dict):
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Config saved to: {config_path}")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def start_training(self):
        self.start_time = time.time()
        self.logger.info("Training started")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_accuracy'].append(train_acc)
        
        msg = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}"
        if val_loss is not None and val_acc is not None:
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_accuracy'].append(val_acc)
            msg += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}"
        
        self.logger.info(msg)
    
    def save_training_curves(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        # Loss plot
        axs[0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        if self.metrics_history['val_loss']:
            axs[0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss')
        axs[0].set_title("Loss over Epochs")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)
        
        # Accuracy plot
        axs[1].plot(epochs, self.metrics_history['train_accuracy'], label='Train Acc')
        if self.metrics_history['val_accuracy']:
            axs[1].plot(epochs, self.metrics_history['val_accuracy'], label='Val Acc')
        axs[1].set_title("Accuracy over Epochs")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        curve_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(curve_path, dpi=300)
        plt.close()
        self.logger.info(f"Training curves saved to {curve_path}")
    
    def finish_training(self):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.logger.info(f"Training completed in {duration:.1f} seconds")
        self.save_training_curves()
