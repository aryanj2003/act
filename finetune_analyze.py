import optuna
import os
import pandas as pd
from plotly.io import show

# Load the Optuna study from the SQLite database
def load_study(ckpt_dir):
    storage_name = f"sqlite:///{ckpt_dir}/study.db"
    print(storage_name)
    study = optuna.load_study(study_name="finetune", storage=storage_name)
    return study

# Load the ranked trials from the Optuna study
def load_ranks(ckpt_dir):
    rank_dir = os.path.join(ckpt_dir, "ranked_trials.csv")
    ranked_trials = pd.read_csv(rank_dir)
    return ranked_trials

# Visualize the optimization history
def visualize_optimization_history(study):
    fig = optuna.visualization.plot_optimization_history(study)
    show(fig)

# Visualize the parameter importances
def visualize_parameter_importances(study):
    fig = optuna.visualization.plot_param_importances(study)
    show(fig)
    
# Visualize the plot rank
def visualize_plot_rank(study, params):
    fig = optuna.visualization.plot_rank(study, params)
    show(fig)
    
# Visualize the slice plot
def visualize_slice_plot(study, params):
    fig = optuna.visualization.plot_slice(study, params)
    show(fig)

ckpt_name = "param_opt_20241023_194521_2000"
ckpt_dir = f"ckpt/{ckpt_name}"
ckpt_dir = "ckpt/param_opt_20241025_132752_2000"
# storage_name = f"sqlite:///{ckpt_dir}/study.db"
# # Study 목록 출력
# study_summaries = optuna.study.get_all_study_summaries(storage=storage_name)
# for study_summary in study_summaries:
#     print(study_summary.study_name)
    
# exit()
study = load_study(ckpt_dir)
# ranked_trials = load_ranks(ckpt_dir)

# print(ranked_trials)


params = ['kl_weight', 'chunk_size', 'batch_size']
visualize_optimization_history(study)
visualize_parameter_importances(study)
visualize_plot_rank(study, params)
visualize_slice_plot(study, params)

