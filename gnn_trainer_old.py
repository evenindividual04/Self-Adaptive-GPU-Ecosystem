import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import wandb

class GNNTrainer:
    def __init__(self, model, config, logger, class_weights=None):
        self.model = model
        self.config = config
        self.logger = logger
        # Prefer Apple MPS -> CUDA -> CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=1e-5
        )
        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.patience = config['training'].get('patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop = False
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0; correct = 0; total = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch if hasattr(data,'batch') else None)
            loss = self.criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs
            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.num_graphs
        return total_loss/total, correct/total
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0; correct = 0; total = 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch if hasattr(data,'batch') else None)
                loss = self.criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs
                preds = out.argmax(dim=1)
                probs = F.softmax(out, dim=1)[:,1]
                correct += (preds == data.y).sum().item()
                total += data.num_graphs
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(data.y.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
        avg_loss = total_loss/total; accuracy = correct/total
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
        }
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metrics['auc'] = 0.0
        return avg_loss, accuracy, metrics, all_labels, all_preds
    '''
    def train(self, train_data, val_data=None):
        train_loader = DataLoader(train_data[0], batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = None
        if val_data:
            val_loader = DataLoader(val_data[0], batch_size=self.config['training']['batch_size'], shuffle=False)
        
        self.logger.start_training()
        epochs = self.config['training']['epochs']
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            
            val_loss, val_acc, val_metrics, _, _ = (0, 0, None, None, None)
            if val_loader:
                val_loss, val_acc, val_metrics, _, _ = self.evaluate(val_loader)
            
            self.logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, "best_model.pth"))
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.logger.info(f"Early stopping triggered at epoch {epoch}")
                    self.early_stop = True
                    break
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_metrics, _, _ = (None, None, None, None, None)
            if val_loader:
                val_loss, val_acc, val_metrics, _, _ = self.evaluate(val_loader)
                self.scheduler.step(val_loss)
            self.logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
            if val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, "best_model.pth"))
            elif val_loader:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.logger.info(f"Early stopping triggered at epoch {epoch}")
                    self.early_stop = True
                    break
        self.logger.finish_training()
        if self.early_stop and val_loader:
            self.model.load_state_dict(torch.load(os.path.join(self.logger.log_dir, "best_model.pth")))
    '''
    
    '''def train(self, train_graphs, val_graphs=None):
        train_loader = DataLoader(train_graphs, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.config['training']['batch_size'], shuffle=False) if val_graphs else None

        self.logger.start_training()
        epochs = self.config['training']['epochs']
        for epoch in range(1, epochs+1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss = None; val_acc = None
            if val_loader:
                val_loss, val_acc, _, _, _ = self.evaluate(val_loader)
                self.scheduler.step(val_loss)
            self.logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            if val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss; self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, "best_model.pth"))
            elif val_loader:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.logger.info(f"Early stopping triggered at epoch {epoch}")
                    self.early_stop = True; break

        self.logger.finish_training()
        if self.early_stop and val_loader:
            self.model.load_state_dict(torch.load(os.path.join(self.logger.log_dir, "best_model.pth"), map_location=self.device))
            '''


    def train(self, train_graphs, val_graphs=None):
        # DataLoaders
        train_loader = DataLoader(
            train_graphs,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_graphs,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        ) if val_graphs else None

        self.logger.start_training()
        epochs = self.config['training']['epochs']

        for epoch in range(1, epochs + 1):
            # One training epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation (if provided)
            val_loss, val_acc = None, None
            if val_loader:
                val_loss, val_acc, _, _, _ = self.evaluate(val_loader)
                # Step LR scheduler on validation loss
                self.scheduler.step(val_loss)

            # Log to your ExperimentLogger
            self.logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            # Log to W&B per-epoch
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            # Early stopping and checkpointing
            if val_loader and val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.logger.log_dir, "best_model.pth")
                    )
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.logger.logger.info(f"Early stopping triggered at epoch {epoch}")
                        self.early_stop = True
                        break

        self.logger.finish_training()

        # Load best checkpoint if early stopped
        if self.early_stop and val_loader:
            self.model.load_state_dict(
                torch.load(os.path.join(self.logger.log_dir, "best_model.pth"), map_location=self.device)
            )

            
            
    '''def final_evaluation(self, test_graphs):
        test_loader = DataLoader(test_graphs, batch_size=self.config['training']['batch_size'], shuffle=False)
        _, _, metrics, y_true, y_pred = self.evaluate(test_loader)
        self.logger.logger.info("Final Test Set metrics:")
        
        for k, v in metrics.items():
            self.logger.logger.info(f" {k.capitalize()}: {v:.4f}")
            
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.tight_layout()
        cm_path = os.path.join(self.logger.log_dir, "confusion_matrix.png")
        plt.savefig(cm_path); plt.close()
        self.logger.logger.info(f"Confusion matrix saved to {cm_path}")'''
        
    
    def final_evaluation(self, test_graphs):
        test_loader = DataLoader(test_graphs, batch_size=self.config['training']['batch_size'], shuffle=False)
        _, _, metrics, y_true, y_pred = self.evaluate(test_loader)
        self.logger.logger.info("Final Test Set metrics:")
        for k, v in metrics.items():
            self.logger.logger.info(f" {k.capitalize()}: {v:.4f}")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.tight_layout()
        cm_path = os.path.join(self.logger.log_dir, "confusion_matrix.png")
        plt.savefig(cm_path); plt.close()
        self.logger.logger.info(f"Confusion matrix saved to {cm_path}")
        return metrics, y_true, y_pred
