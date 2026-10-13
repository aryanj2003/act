import subprocess
import optuna
import pandas as pd
import numpy as np
import os
from datetime import datetime


"""_summary_

This script is used to optimize the hyperparameters of the ACT model using Optuna.

"""

# Set parameters
num_trials = 1 # Number of trials with different hyperparameters
num_repeats = 1 # Number of times to repeat the for each trial
optimize_direction = "maximize" # maximize or minimize

EPOCH = 2000

# Get date and time for the study directory
STUDY_DIR = f"ckpt/param_opt_mamba_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{EPOCH}"

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter ranges
    # From "ACT Tuning Tips -- https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit"
    
    kl_weight = trial.suggest_int('kl_weight', 90, 90, step=10)
    chunk_size = trial.suggest_int('chunk_size', 160, 160, step=10)
    batch_size = trial.suggest_int('batch_size', 5, 5, step=1)
    # hidden_dim = trial.suggest_int('hidden_dim', 256, 512, step=64)
    # dim_feedforward = trial.suggest_int('dim_feedforward', 1000, 4000, step=500)
    learning_rate = 1e-5 * batch_size / 2
    # learning_rate = 1e-5 # Just checking for the best hyperparameter suggested by the paper
    # learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)

    # Set ckpt directory
    ckpt_dir = f"{STUDY_DIR}/{trial.number}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    
    # ckpt_dir = "ckpt/test"
    
    # Create the command for training the model
    train_command = [
        "python3", "imitate_episodes_mamba_param_opt.py",
        "--task_name", "sim_transfer_cube_scripted",
        "--ckpt_dir", str(ckpt_dir),  # Replace with your checkpoint directory
        "--policy_class", "MAMBA2",
        "--kl_weight", str(kl_weight),
        "--chunk_size", str(chunk_size),
        "--hidden_dim", "512",
        "--batch_size", str(batch_size),
        "--dim_feedforward", "3200",
        "--num_epochs", str(EPOCH),
        "--lr", str(learning_rate),
        "--seed", "0"
    ]

    # Create the command for evaluating the model
    eval_command = train_command.copy()
    eval_command.append("--temporal_agg")
    eval_command.append("--eval")

    evaluation_metrics = []
    for idx in range(num_repeats):
        # Run the training command
        subprocess.run(train_command)
        
        # Run the evaluation command
        result = subprocess.run(eval_command, capture_output=True, text=True)
        
        # Parse the evaluation result (success rate) - customized only for ACT model script
        output_lines = result.stdout.splitlines()
        success_rate = float(output_lines[-1])
        print(success_rate)
        evaluation_metric = float(success_rate)
        
        # Get statistics and log them
        evaluation_metrics.append(evaluation_metric)
        mean = np.mean(evaluation_metrics)
        std  = np.std(evaluation_metrics)
        
        trial.set_user_attr('mean', mean)
        trial.set_user_attr('std', std)
        
        print(f"Trial {trial.number} | Repetition {idx} - Success Rate: {success_rate} | Mean: {mean} | Std: {std}")
        
    return mean # Optimizes using the mean of the success rates

def rank_trials(study): 
    # Create a DataFrame with the trial results
    trials_data = [{
        'Trial Number': trial.number,
        'kl_weight': trial.params['kl_weight'],
        'chunk_size': trial.params['chunk_size'],
        'hidden_dim': '512',
        'batch_size': trial.params['batch_size'],
        'dim_feedforward': '3200',
        'Success Rates (avg)': trial.value,
        'Standard Deviation': trial._user_attrs['std'],
        'Number of Repetiions': num_repeats
    } for trial in study.trials]

    df_trials = pd.DataFrame(trials_data)

    # Sort the trials by the Success Rates (descending order for maximizing)
    df_sorted = df_trials.sort_values(by='Success Rates (avg)', ascending=True)

    # Display the sorted DataFrame
    return df_sorted

# Set directory for Optuna study
os.makedirs(STUDY_DIR, exist_ok=True)

storage_name = f"sqlite:///{STUDY_DIR}/study.db"

# Run the optimization
study = optuna.create_study(direction=optimize_direction, study_name = 'finetune', storage=storage_name, load_if_exists=False)  
study.optimize(objective, n_trials=num_trials)

# Rank the trials based on the Success Rates and display them
ranked_trials = rank_trials(study)

# Save the ranked trials to a CSV file
ranked_trials.to_csv(f"{STUDY_DIR}/ranked_trials.csv", index=False)
print(ranked_trials)
