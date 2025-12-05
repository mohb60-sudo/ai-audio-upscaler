import os
import optuna
import torch
import threading
from train import train_model

# Global stop event for tuning
tuning_stop_event = threading.Event()

def objective(trial):
    """
    Optuna objective function.
    """
    # Define search space
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
    num_layers = trial.suggest_categorical("num_layers", [3, 4, 5])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    
    # Get data dir from global args (or pass via study.set_user_attr)
    data_dir = trial.study.user_attrs["data_dir"]
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing Hyperparameters")
    print(f"  - base_channels: {base_channels}")
    print(f"  - num_layers: {num_layers}")
    print(f"  - lr: {lr:.6f}")
    print(f"  - batch_size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Use a unique save path for each trial
    # We use a temp directory or a specific tuning directory
    tune_dir = os.path.join(os.getcwd(), "tune_results")
    os.makedirs(tune_dir, exist_ok=True)
    save_path = os.path.join(tune_dir, f"model_trial_{trial.number}.ckpt")
    
    try:
        # Check if user requested stop
        if tuning_stop_event.is_set():
            print(f"\n⏹️  Trial {trial.number} Stopped by User\n")
            raise optuna.exceptions.TrialPruned()
        
        # Run training (SHORT run for tuning)
        result = train_model(
            data_dir=data_dir,
            save_path=save_path,
            epochs=3,  # Reduced from 5 for faster tuning
            batch_size=batch_size,
            lr=lr,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_gan=False,  # Disable GAN for faster tuning
            base_channels=base_channels,
            num_layers=num_layers,
            num_workers=0,  # CRITICAL: Set to 0 for Windows stability
            use_amp=True,
            stop_event=tuning_stop_event  # Support stop button
        )
        
        # Report metric to Optuna
        # We want to minimize LSD
        final_lsd = result["final_val_lsd"]
        print(f"\n✓ Trial {trial.number} Complete: LSD = {final_lsd:.4f}\n")
        return final_lsd
        
    except Exception as e:
        print(f"\n✗ Trial {trial.number} Failed: {str(e)}\n")
        # Prune failed trials
        raise optuna.exceptions.TrialPruned()

def run_tuning_session(data_dir, num_trials=10, progress_callback=None):
    """
    Runs an Optuna tuning session.
    
    Args:
        data_dir (str): Path to dataset.
        num_trials (int): Number of trials to run.
        progress_callback (callable): Optional callback (trial_number, best_value) -> None.
        
    Returns:
        optuna.study.Study: The completed study.
    """
    # Create study
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("data_dir", data_dir)
    
    # Determine parallel jobs based on GPU availability
    n_jobs = 1  # Default: sequential
    if torch.cuda.is_available():
        try:
            # Check available VRAM
            torch.cuda.empty_cache()
            free_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Conservative estimate: need ~4GB per trial
            # Allow parallel if >8GB total VRAM
            if free_vram_gb > 8.0:
                n_jobs = 2
                print(f"\n💡 GPU VRAM: {free_vram_gb:.1f}GB detected")
                print("   Enabling PARALLEL trials (n_jobs=2) for ~1.8x speedup\n")
            else:
                print(f"\n💡 GPU VRAM: {free_vram_gb:.1f}GB detected")
                print("   Running SEQUENTIAL trials (n_jobs=1) to avoid OOM\n")
        except Exception as e:
            print(f"\n⚠️  Could not detect VRAM: {e}")
            print("   Running SEQUENTIAL trials (n_jobs=1) for safety\n")
    
    # Define a callback adapter for Optuna
    callbacks = []
    if progress_callback:
        def adapter(study, trial):
            try:
                best_val = study.best_value
            except ValueError:
                best_val = float('inf')
            progress_callback(len(study.trials), best_val)
        callbacks.append(adapter)
    
    study.optimize(objective, n_trials=num_trials, n_jobs=n_jobs, callbacks=callbacks)
    
    return study
