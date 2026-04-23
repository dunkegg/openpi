import dataclasses

# import jax

# from openpi.models import model as _model
# from openpi.policies import droid_policy
# from openpi.policies import policy_config as _policy_config
# from openpi.shared import download
# from openpi.training import config as _config
# from openpi.training import data_loader as _data_loader

# config = _config.get_config("pi0_base")
# # checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
# checkpoint_dir = "checkpoints/pi0_base"

# # Create a trained policy.
# policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
# example = droid_policy.make_droid_example()
# result = policy.infer(example)

# # Delete the policy to free up memory.
# del policy

# print("Actions shape:", result["actions"].shape)


# eval_pi0_metrics.py
# 用训练好的 OpenPI checkpoint 在原数据集上跑预测，并计算 MAE / RMSE

import dataclasses
import numpy as np
from tqdm import tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.policies import policy_config as _policy_config
import jax

# =====================================================
# 修改这里
# =====================================================
CONFIG_NAME = "pi05_wh1"
CHECKPOINT_DIR = "checkpoints/pi05_wh1/test_express_1/80000"
NUM_BATCHES = 50          # 先小一点测试
BATCH_SIZE = 4
# =====================================================

def to_numpy(x):
    return jax.tree_util.tree_map(lambda v: np.array(v), x)

def main():
    # ----------------------------
    # load config
    # ----------------------------
    config = _config.get_config(CONFIG_NAME)
    config = dataclasses.replace(config, batch_size=BATCH_SIZE)

    print("Loading policy...")
    policy = _policy_config.create_trained_policy(
        config,
        CHECKPOINT_DIR,
    )

    print("Loading dataloader...")
    loader = _data_loader.create_data_loader(
        config,
        num_batches=NUM_BATCHES,
        skip_norm_stats=False,
    )

    mae_list = []
    mse_list = []

    print("Start evaluating...")
    count = 0
    for obs, gt in tqdm(loader):
        count+=1
        if count>10:
            break
        # --------------------------------
        # gt action
        # --------------------------------
        gt = to_numpy(gt)
        if isinstance(gt, dict):
            gt_action = np.asarray(gt["actions"])
        else:
            gt_action = np.asarray(gt)

        # --------------------------------
        # pred action
        # --------------------------------
        obs_dict = {
        "images": {
            "cam_low": np.asarray(obs.images["base_0_rgb"]),
            "cam_left_wrist": np.asarray(obs.images["left_wrist_0_rgb"]),
            "cam_right_wrist": np.asarray(obs.images["right_wrist_0_rgb"]),
        },
        "state": np.asarray(obs.state),
    }
        pred = policy.infer(obs_dict)

        # 看返回字段名
        if "actions" in pred:
            pred_action = np.asarray(pred["actions"])
        elif "action" in pred:
            pred_action = np.asarray(pred["action"])
        else:
            print("pred keys =", pred.keys())
            raise ValueError("Cannot find actions in prediction.")

        # --------------------------------
        # 对齐 shape
        # --------------------------------
        min_b = min(gt_action.shape[0], pred_action.shape[0])

        gt_action = gt_action[:min_b]
        pred_action = pred_action[:min_b]

        # horizon 对齐
        if gt_action.ndim == 3 and pred_action.ndim == 3:
            min_h = min(gt_action.shape[1], pred_action.shape[1])
            gt_action = gt_action[:, :min_h]
            pred_action = pred_action[:, :min_h]

        # dim 对齐
        min_d = min(gt_action.shape[-1], pred_action.shape[-1])
        gt_action = gt_action[..., :min_d]
        pred_action = pred_action[..., :min_d]

        err = pred_action - gt_action

        mae_list.append(np.abs(err))
        mse_list.append(err ** 2)

    # --------------------------------
    # metrics
    # --------------------------------
    mae = np.concatenate(mae_list).mean()
    rmse = np.sqrt(np.concatenate(mse_list).mean())

    print("\n======================")
    print("MAE  =", float(mae))
    print("RMSE =", float(rmse))
    print("======================\n")


if __name__ == "__main__":
    main()