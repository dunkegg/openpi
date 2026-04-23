import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_wh1_example() -> dict:
    """Creates a random input example for the wh1 policy."""
    return {
        "state": np.ones((26,)),
        "images": {
            # "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

@dataclasses.dataclass(frozen=True)
class WH1Inputs(transforms.DataTransformFn):

    """Inputs for the WH1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [26]
    - actions: [action_horizon, 26]
    """

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        # print("+++++++++++++++++++++++++++++")
        # print("cam_low type:", type(in_images["cam_low"]))
        # print("cam_low shape:", getattr(in_images["cam_low"], "shape", None))
        # print("cam_low dtype:", getattr(in_images["cam_low"], "dtype", None))
        base_image =  _to_numpy_image(in_images["cam_low"])

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] =  _to_numpy_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        state = np.asarray(data["state"], dtype=np.float32)
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }


        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class WH1Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 26 dims.
        return {"actions": np.asarray(data["actions"][:, :26])}
    
def _to_numpy_image(x):
    import torch
    import numpy as np

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # CHW -> HWC
    if x.shape[0] in [1,3] and x.ndim == 3:
        x = np.transpose(x, (1, 2, 0))

    # float -> uint8
    if x.dtype != np.uint8:
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    return x
def _parse_image(image):
    image = np.asarray(image)

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image