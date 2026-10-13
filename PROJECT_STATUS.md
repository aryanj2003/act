# ACT Project Status

**Date: 2026-10-12**

## 1. ACT codebase

This work started from the original ACT repository. I kept the original `main` branch separate and created my own branch:

- GitHub fork: `aryanj2003/act`
- Working branch: `aryanj2003/aloha-act-work`
- Main source commit: `9acf07b`
- Offline plot commit: `89f3aa9`

I looked through the main ACT files and the ALOHA robot files. The important parts are:

- `imitate_episodes.py`: trains ACT and runs evaluation.
- `policy.py`: builds and calls the ACT policy.
- `detr/`: contains the ACT model code.
- `utils.py`: loads HDF5 data and calculates normalization statistics.
- `data/cube_pp/`: contains the recorded robot episodes.
- `ckpt/`: contains local checkpoints and training results.
- `aloha_scripts/`: contains the real robot, teleoperation, recording, replay, and sleep scripts.

I also added `offline_check.py`. It loads one recorded episode, loads the trained checkpoint, runs the policy on the recorded observations, and makes plots comparing recorded actions with predicted actions.

The offline test was run on episode 70. The two checkpoints were compared:

- `policy_best.ckpt`: MAE `0.028585`, RMSE `0.043147`
- `policy_last.ckpt`: MAE `0.031924`, RMSE `0.048533`

Because the best checkpoint had lower errors, `policy_best.ckpt` was selected.

## 2. Software and data structure investigation

The training data is stored in HDF5 files. Each episode has:

- robot joint positions in `/observations/qpos`
- camera images in `/observations/images/<camera_name>`
- actions in `/action`

The policy uses the saved `dataset_stats.pkl` file to normalize qpos and un-normalize predicted actions. The ACT model was built with the same training settings: chunk size 100, hidden size 512, feedforward size 3200, KL weight 10, and 2 cameras for `real_cube_pp`.

The original ACT model builder uses `argparse` while it builds the model. Because of this, `offline_check.py` passes the same training flags when it creates the model. The extra offline-check arguments are removed before the ACT model parser sees them.

Problems found during the work:

- The base `miniconda3` environment did not have PyTorch. The `aloha` environment had the needed PyTorch and CUDA setup.
- The first offline checker version passed extra arguments into the ACT parser. This was fixed.
- The checker initially stopped after one timestep for testing. It was later changed to run a full episode.
- The real robot evaluation loaded `policy_best.ckpt` correctly, but ROS could not find the robot drivers.
- `roscore` was needed before starting the robot launch.
- The `xs_sdk` driver started but could not find any Dynamixel motor IDs. It reported “There is no status packet.”
- Starting only one puppet robot did not fix the problem.
- The USB adapters were visible, but `lsusb -t` showed all four robot adapters under one USB hub. The ALOHA README recommends separate USB ports for the robot adapters.
- Dynamixel Wizard also did not show detected motors after scanning, so the problem is below ACT and below normal ROS policy code.
- The ALOHA README also mentions a local `arm.py` change to reduce teleoperation delay. That change has not been treated as the cause of the motor detection failure.

The generated offline checker file and its three plots were pushed to the personal fork. Large checkpoints and local videos were left out of GitHub because they are too large for normal GitHub storage.

## 3. Next steps

The real robot test is currently paused. The next step is for the PhD student to check the Dynamixel communication setup. In Dynamixel Wizard, the puppet ports should be scanned with Protocol 2.0, baud rate 1,000,000, and IDs 0 to 10. Each puppet robot should show about 9 motors.

If the motors appear in Wizard, the ROS port or communication settings can be checked again. If they do not appear, the issue is probably with the robot bus, adapter, power, or communication settings. The ACT policy should not be run on the robot until the `xs_sdk` drivers start without motor detection errors.
