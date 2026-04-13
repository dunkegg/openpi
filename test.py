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

from datasets import load_dataset

# 指定下载到 /mnt/data/libero_dataset
dataset = load_dataset(
    "openvla/modified_libero_rlds",
    cache_dir="data/libero_dataset"
)