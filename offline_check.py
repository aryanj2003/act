import argparse
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from imitate_episodes import make_policy


def load_episode(dataset_dir, episode_idx, camera_names):
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        action = root['/action'][()]
        images = {
            camera_name: root[f'/observations/images/{camera_name}'][()]
            for camera_name in camera_names
        }
    return qpos, action, images


def get_task_config(task_name):
    if task_name.startswith('sim_'):
        from constants import SIM_TASK_CONFIGS
        return SIM_TASK_CONFIGS[task_name]
    from aloha_scripts.constants import TASK_CONFIGS
    return TASK_CONFIGS[task_name]


def main(args):
    task_config = get_task_config(args.task_name)
    dataset_dir = task_config['dataset_dir']
    camera_names = task_config['camera_names']

    with open(os.path.join(args.ckpt_dir, 'dataset_stats.pkl'), 'rb') as stats_file:
        stats = pickle.load(stats_file)

    qpos_all, action_all, images = load_episode(
        dataset_dir, args.episode_idx, camera_names
    )
    if len(qpos_all) != len(action_all):
        raise ValueError('The episode qpos and action arrays have different lengths')

    policy_config = {
        'lr': args.lr,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
    }
    model_argv = [sys.argv[0]]
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in ('--episode_idx', '--checkpoint'):
            skip_next = True
            continue
        model_argv.append(argument)
    checker_argv = sys.argv
    sys.argv = model_argv
    policy = make_policy(args.policy_class, policy_config)
    sys.argv = checker_argv
    checkpoint_path = os.path.join(args.ckpt_dir, args.checkpoint)
    loading_status = policy.load_state_dict(torch.load(checkpoint_path))
    print(loading_status)
    policy.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device)
    qpos_mean = stats['qpos_mean']
    qpos_std = stats['qpos_std']
    action_mean = stats['action_mean']
    action_std = stats['action_std']

    predicted_actions = []
    with torch.inference_mode():
        for timestep in range(len(qpos_all)):
            qpos = (qpos_all[timestep] - qpos_mean) / qpos_std 
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
            
            # TODO 1: normalize qpos_all[t] with stats -> float tensor on cuda, shape (1, 14). 

            curr_images = []
            for cam_name in camera_names:
                curr_img = np.transpose(images[cam_name][timestep], (2, 0, 1))
                
                curr_images.append(curr_img) 
                
            curr_img = np.stack(curr_images, axis=0)
            curr_img = torch.from_numpy(curr_img/255.0).float().cuda().unsqueeze(0)
            
        
            # TODO 2: build the image tensor, shape (1, num_cams, 3, H, W), floats in [0, 1] and then on cuda.
            # (look at get_image(); here images[c][t] is (H, W, 3))

            action_chunk = policy(qpos, curr_img)

            # Take the first action from the predicted chunk.
            a0 = action_chunk[0, 0].detach().cpu().numpy()
            
            # Un-normalize the action.
            a0 = a0 * action_std + action_mean

            # TODO 3: take the FIRST action of the chunk and un-normalize it -> numpy array (14, ).
            predicted_actions.append(a0)

    predicted_actions = np.asarray(predicted_actions)
    mae = np.mean(np.abs(action_all - predicted_actions))
    rmse = np.sqrt(np.mean((action_all - predicted_actions) ** 2))
    print(f'{args.checkpoint}: MAE={mae:.6f}, RMSE={rmse:.6f}')
    checkpoint_stem = os.path.splitext(args.checkpoint)[0]
    plot_path = f'offline_check_ep{args.episode_idx}_{checkpoint_stem}.png'
    figure, axes = plt.subplots(14, 1, figsize=(12, 28), sharex=True)
    for dimension, axis in enumerate(axes):
        axis.plot(action_all[:, dimension], label='recorded action')
        axis.plot(predicted_actions[:, dimension], label='predicted action')
        axis.set_ylabel(f'Action {dimension}')
        axis.grid(True, alpha=0.25)
        axis.legend(loc='upper right')
    axes[-1].set_xlabel('Timestep')
    figure.suptitle(f'ACT offline reproduction, episode {args.episode_idx}')
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--policy_class', required=True)
    parser.add_argument('--kl_weight', type=int, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--dim_feedforward', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--episode_idx', type=int, required=True)
    parser.add_argument('--checkpoint', default='policy_best.ckpt')
    main(parser.parse_args())