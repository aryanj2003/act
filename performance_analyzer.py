import subprocess
import numpy as np
import pandas as pd
import os
import ast
from collections import Counter
from sim_env import make_sim_env

"""
Aggregates success rates (or returns) for each task, both in simulation and real.
Shows number of successful trials for each subtask from a given task, among N seeds and K policy evaluations (rollouts).

No. of rollouts: You may configure it in the "imitate_episodes*.py" script, "num_rollouts" parameter.
Intended to reproduce the results of Table 1 & 2 in the original ALOHA paper (https://arxiv.org/pdf/2304.13705).

"""

num_seeds = 3
is_temporal_agg = True

task = "sim_transfer_cube_scripted"

# Policy
policy_class = "ACT"
if policy_class == "MAMBA2":
    dir_rollout_script = "imitate_episodes_mamba_param_opt.py"
    ckpt_dir = "ckpt/param_opt_mamba_20241109_050950_2000/0"
elif policy_class == "ACT": 
    dir_rollout_script = "imitate_episodes_param_opt.py"
    ckpt_dir = "ckpt/test"
else:
    raise NotImplementedError

# Model hyperparameters only for evaluation
# kl_weight = 90
# chunk_size = 160
# batch_size = 5

# ACT
kl_weight = 10
chunk_size = 100
batch_size = 16
# learning_rate = 1e-5 * batch_size / 2
learning_rate = 2e-5
# Get maximum reward possible for the given task (total number of subtasks)
env = make_sim_env(task)
env_max_reward = env.task.max_reward

# Terminal command for policy evaluation
eval_command = [
        "python3", dir_rollout_script,
        "--task_name", "sim_transfer_cube_scripted",
        "--ckpt_dir", ckpt_dir, 
        "--policy_class", policy_class,
        "--kl_weight", str(kl_weight),
        "--chunk_size", str(chunk_size),
        "--hidden_dim", "512",
        "--batch_size", str(batch_size),
        "--dim_feedforward", "3200",
        "--num_epochs", "2000",
        "--lr", str(learning_rate),
        "--eval"
    ]

if is_temporal_agg:
    eval_command.append("--temporal_agg")

# Insert default seed
eval_command.append("--seed")
eval_command.append("0")

# Collect action rewards from N different random seeds
rewards = []
for _ in range(num_seeds):
    
    # Random seed
    seed = np.random.randint(0,1000)
    
    eval_command = eval_command[:-1]
    eval_command.append(str(seed))
    
    subprocess.run(eval_command)
    
    with open(os.path.join(ckpt_dir, "result_policy_best.txt"), 'r') as f:
        
        eval_summary = f.readlines()
        rewards_per_seed = ast.literal_eval(eval_summary[-1])
        rewards_per_seed = [int(i) for i in rewards_per_seed]
        
        rewards += rewards_per_seed
        
# Number of rewards per reward value for the entire rollouts
count_dict = Counter(rewards)
counts_by_rewards = pd.DataFrame(list(count_dict.items()), columns = ["Reward", "Count"])

max_reward = env_max_reward * np.ones(len(counts_by_rewards))
counts_by_rewards_ranked = counts_by_rewards.sort_values(by="Reward").reset_index(drop=True)

# Success rates (%) per subtasks. Subtask IDs(or indices) correspond to reward values
df = counts_by_rewards_ranked.copy()

for subtask_idx in range(1, len(df)):
    trial_counts = df.iloc[subtask_idx,1]
    df.iloc[0:subtask_idx,1] += trial_counts

num_trials = df["Count"][0] # Total number of trials
df = df.iloc[1:] # Get rid of the first row since it just stands for "Start of the task"
df.rename(columns={"Count" : "Success Rate (%)"}, inplace=True)
df.iloc[:,1]  = df.iloc[:,1] / num_trials * 100 # Get success rates
success_rates_per_subtask = df

# Add Maximum reward / subtask ID for readability
counts_by_rewards_ranked["Maximum Reward"] = max_reward
success_rates_per_subtask["Maximum Subtask Index"] = max_reward[:-1]
success_rates_per_subtask.rename(columns={"Reward":"Subtask"}, inplace=True)

print(counts_by_rewards_ranked)
print(success_rates_per_subtask)

# Save both tables as csv 
counts_by_rewards_ranked.to_csv(os.path.join(ckpt_dir, f"successes_by_rewards_{num_seeds}seeds.csv"), index=False)
success_rates_per_subtask.to_csv(os.path.join(ckpt_dir, f"success_rates_per_subtask_{num_seeds}seeds.csv"), index=False)
