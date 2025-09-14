# optuna_tune.py (optimized)

import optuna
from train_sim import main as train_main  # Assumes train_sim.py is in same dir

def objective(trial):
    config = {
        "model": {
            "input_dim": 9,  # Fixed to match your features (5 telemetry + 4 NX)
            "output_dim": 2,
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),  # Focused options
            "dropout": trial.suggest_float("dropout", 0.3, 0.5, step=0.05),  # Narrowed for precision
            "heads": trial.suggest_int("heads", 4, 8),
            "dropedge_p": trial.suggest_float("dropedge_p", 0.05, 0.2, step=0.05)  # Narrowed
        },
        "training": {
            "learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),  # Log scale for LR
            "batch_size": 32,  # Fixed for stability (tune if memory varies)
            "epochs": 80,
            "patience": 10
        }
    }
    wandb_run_name = f"trial_{trial.number}"
    test_f1 = train_main(config=config, wandb_run_name=wandb_run_name)
    return test_f1

if __name__ == "__main__":
    # Optional: Use storage for resuming interrupted studies
    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db", study_name="gnn_tuning", load_if_exists=True)
    study.optimize(objective, n_trials=20)  # Tune n_trials (15-25 recommended for time)
    print("Best params:", study.best_params, "Best F1:", study.best_value)
