#!/usr/bin/env python3
"""
Convert `output` dataset into a jean_auto_force_viz-like v3 dataset layout.

Input layout (example):
  output/
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.chest_rgb/episode_000000.mp4
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl

Output layout:
  <dst>/
    data/chunk-000/file-000.parquet
    videos/observation.images.chest/chunk-000/file-000.mp4
    videos/observation.images.left/chunk-000/file-000.mp4
    videos/observation.images.right/chunk-000/file-000.mp4
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_VIDEO_TARGETS = (
    ("observation.images.chest", "observation.images.chest_rgb"),
    ("observation.images.left", "observation.images.left_wrist_rgb"),
    ("observation.images.right", "observation.images.right_wrist_rgb"),
)
VIDEO_SOURCE_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
DEFAULT_TASK_TEXT = (
    "Pick up a package with your left hand, then hand it to your right hand, "
    "and use your right hand to place the package on the conveyor belt."
)

VECTOR_DIMS = {
    "observation.state.left_tcp": 7,
    "observation.state.right_tcp": 7,
    "observation.state.left_delta_tcp": 6,
    "observation.state.right_delta_tcp": 6,
    "observation.state.left_finger_pressure": 6,
    "observation.state.right_finger_pressure": 6,
    "observation.state.left_wrist_force": 3,
    "observation.state.right_wrist_force": 3,
    "action.left_delta_tcp": 6,
    "action.right_delta_tcp": 6,
}

SCALAR_FLOAT_FEATURES = [
    "observation.state.left_pinch",
    "observation.state.right_pinch",
    "action.left_pinch",
    "action.right_pinch",
    "task.progress.chest",
    "task.progress.left",
    "task.progress.right",
    "timestamp",
]

SCALAR_INT_FEATURES = [
    "teleoperated",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]

CONVERTED_COLUMN_ORDER = [
    "observation.state.left_tcp",
    "observation.state.right_tcp",
    "observation.state.left_delta_tcp",
    "observation.state.right_delta_tcp",
    "observation.state.left_pinch",
    "observation.state.right_pinch",
    "observation.state.left_finger_pressure",
    "observation.state.right_finger_pressure",
    "observation.state.left_wrist_force",
    "observation.state.right_wrist_force",
    "action.left_delta_tcp",
    "action.right_delta_tcp",
    "action.left_pinch",
    "action.right_pinch",
    "task.progress.chest",
    "task.progress.left",
    "task.progress.right",
    "teleoperated",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]

STATS_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]

LEFT_TCP_NAMES = [
    "end_position_l_x",
    "end_position_l_y",
    "end_position_l_z",
    "end_quaternion_l_x",
    "end_quaternion_l_y",
    "end_quaternion_l_z",
    "end_quaternion_l_w",
]
RIGHT_TCP_NAMES = [
    "end_position_r_x",
    "end_position_r_y",
    "end_position_r_z",
    "end_quaternion_r_x",
    "end_quaternion_r_y",
    "end_quaternion_r_z",
    "end_quaternion_r_w",
]
# Keep the existing pinch definition for now: four MCP joints excluding thumb.
LEFT_PINCH_NAMES = [
    "INDEX_MCP_LEFT",
    "MIDDLE_MCP_LEFT",
    "RING_MCP_LEFT",
    "LITTLE_MCP_LEFT",
]
RIGHT_PINCH_NAMES = [
    "INDEX_MCP_RIGHT",
    "MIDDLE_MCP_RIGHT",
    "RING_MCP_RIGHT",
    "LITTLE_MCP_RIGHT",
]
LEFT_FINGER_PRESSURE_NAMES = [
    "THUMB_MP_LEFT_PRESSURE",
    "THUMB_CMC_LEFT_PRESSURE",
    "INDEX_MCP_LEFT_PRESSURE",
    "MIDDLE_MCP_LEFT_PRESSURE",
    "RING_MCP_LEFT_PRESSURE",
    "LITTLE_MCP_LEFT_PRESSURE",
]
RIGHT_FINGER_PRESSURE_NAMES = [
    "THUMB_MP_RIGHT_PRESSURE",
    "THUMB_CMC_RIGHT_PRESSURE",
    "INDEX_MCP_RIGHT_PRESSURE",
    "MIDDLE_MCP_RIGHT_PRESSURE",
    "RING_MCP_RIGHT_PRESSURE",
    "LITTLE_MCP_RIGHT_PRESSURE",
]
LEFT_WRIST_FORCE_NAMES = [
    "WRIST_FORCE_LEFT_X",
    "WRIST_FORCE_LEFT_Y",
    "WRIST_FORCE_LEFT_Z",
]
RIGHT_WRIST_FORCE_NAMES = [
    "WRIST_FORCE_RIGHT_X",
    "WRIST_FORCE_RIGHT_Y",
    "WRIST_FORCE_RIGHT_Z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert output dataset to jean-like v3 layout.")
    parser.add_argument("--src", type=Path, default=Path("output"), help="Source dataset root.")
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("output_jean_like"),
        help="Target dataset root to create.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=224,
        help="Output video width (default: 224).",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=224,
        help="Output video height (default: 224).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="chunks_size value to write into meta/info.json.",
    )
    parser.add_argument(
        "--task-text",
        type=str,
        default=DEFAULT_TASK_TEXT,
        help="Task text written into meta/tasks.parquet and meta/episodes tasks.",
    )
    parser.add_argument(
        "--video-map",
        action="append",
        default=[],
        metavar="SRC_KEY:DST_KEY",
        help=(
            "Map source video key to target key. "
            "Example: observation.images.chest_rgb:observation.images.chest or "
            "observation.images.left_wrist_rgb:observation.images.left,observation.images.right. "
            "Can be provided multiple times. If omitted, defaults to chest_rgb->chest, "
            "left_wrist_rgb->left, and right_wrist_rgb->right. If right_wrist_rgb is missing, "
            "the script falls back to duplicating left_wrist_rgb into the right camera."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination directory if it already exists.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_video_map(video_map_args: Iterable[str]) -> List[Tuple[str, str]]:
    mapping: List[Tuple[str, str]] = []
    for item in video_map_args:
        if ":" not in item:
            raise ValueError(f"Invalid --video-map value `{item}`, expected SRC_KEY:DST_KEY[,DST_KEY...]")
        src_key, dst_keys = item.split(":", 1)
        src_key = src_key.strip()
        dst_key_list = [dst_key.strip() for dst_key in dst_keys.split(",") if dst_key.strip()]
        if not src_key or not dst_key_list:
            raise ValueError(f"Invalid --video-map value `{item}`, expected SRC_KEY:DST_KEY[,DST_KEY...]")
        for dst_key in dst_key_list:
            mapping.append((src_key, dst_key))
    return mapping


def resolve_default_video_map(src_info: Dict[str, Any]) -> List[Tuple[str, str]]:
    src_features = src_info.get("features", {})
    available = set(src_features)

    mapping: List[Tuple[str, str]] = []
    for dst_key, preferred_src_key in DEFAULT_VIDEO_TARGETS:
        if preferred_src_key in available:
            mapping.append((preferred_src_key, dst_key))
            continue

        if dst_key == "observation.images.right" and "observation.images.left_wrist_rgb" in available:
            print(
                "Warning: source dataset is missing observation.images.right_wrist_rgb; "
                "duplicating observation.images.left_wrist_rgb into observation.images.right."
            )
            mapping.append(("observation.images.left_wrist_rgb", dst_key))
            continue

        raise KeyError(
            f"Missing required source video key '{preferred_src_key}' while building default video mapping. "
            f"Available video keys: {sorted(k for k in available if k.startswith('observation.images.'))}"
        )

    return mapping


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=1,
    )


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm[q_norm < 1e-12] = 1.0
    q = q / q_norm

    # Choose the shortest rotation.
    sign = np.where(q[:, 3:4] < 0.0, -1.0, 1.0)
    q *= sign

    v = q[:, :3]
    w = np.clip(q[:, 3], -1.0, 1.0)
    nv = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(nv, w)

    rotvec = np.zeros_like(v)
    small = nv < 1e-8
    rotvec[small] = 2.0 * v[small]
    not_small = ~small
    if np.any(not_small):
        axis = v[not_small] / nv[not_small, None]
        rotvec[not_small] = axis * angle[not_small, None]
    return rotvec


def compute_delta_tcp(obs_tcp: np.ndarray, act_tcp: np.ndarray) -> np.ndarray:
    obs_pos = obs_tcp[:, :3]
    obs_q = obs_tcp[:, 3:7]
    act_pos = act_tcp[:, :3]
    act_q = act_tcp[:, 3:7]

    delta_pos = act_pos - obs_pos
    obs_q_conj = np.concatenate([-obs_q[:, :3], obs_q[:, 3:4]], axis=1)
    q_delta = quat_mul(act_q, obs_q_conj)
    delta_rot = quat_to_rotvec(q_delta)
    return np.concatenate([delta_pos, delta_rot], axis=1)


def to_fixed_size_list(arr2d: np.ndarray, value_type: pa.DataType) -> pa.Array:
    arr2d = np.asarray(arr2d)
    if arr2d.ndim != 2:
        raise ValueError(f"Expected 2D array for fixed-size list, got shape={arr2d.shape}")
    flat = pa.array(arr2d.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, arr2d.shape[1])


def summarize_array(values: np.ndarray) -> Dict[str, List[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def summarize_video_placeholder(frame_count: int, channels: int) -> Dict[str, Any]:
    # LeRobot expects image/video stats tensors like (C,1,1), e.g. (3,1,1).
    zeros = [[[0.0]] for _ in range(channels)]
    return {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": zeros,
        "count": [int(frame_count)],
        "q01": zeros,
        "q10": zeros,
        "q50": zeros,
        "q90": zeros,
        "q99": zeros,
    }


def column_to_2d_float(table: pa.Table, column_name: str, row_count: int) -> np.ndarray:
    if column_name not in table.column_names:
        return np.zeros((row_count, 0), dtype=np.float32)
    values = table.column(column_name).to_pylist()
    if not values:
        return np.zeros((row_count, 0), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def extract_slice_with_zeros(arr: np.ndarray, start: int, end: int, row_count: int) -> np.ndarray:
    width = end - start
    out = np.zeros((row_count, width), dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2:
        return out
    if start >= arr.shape[1]:
        return out
    src_end = min(end, arr.shape[1])
    out[:, : src_end - start] = arr[:, start:src_end]
    return out


def mean_indices_with_zeros(arr: np.ndarray, indices: List[int], row_count: int) -> np.ndarray:
    if arr.size == 0 or arr.ndim != 2:
        return np.zeros(row_count, dtype=np.float32)
    valid = [idx for idx in indices if idx < arr.shape[1]]
    if not valid:
        return np.zeros(row_count, dtype=np.float32)
    return arr[:, valid].mean(axis=1).astype(np.float32)


def get_scalar_column_or_default(
    table: pa.Table,
    column_name: str,
    row_count: int,
    dtype: np.dtype,
    default_value: float | int = 0,
) -> np.ndarray:
    if column_name not in table.column_names:
        return np.full(row_count, default_value, dtype=dtype)
    return np.asarray(table.column(column_name).to_pylist(), dtype=dtype)


def get_feature_names(src_info: Dict[str, Any], feature_key: str) -> List[str]:
    feature = src_info.get("features", {}).get(feature_key)
    if feature is None:
        raise KeyError(f"Missing feature '{feature_key}' in source meta/info.json")
    names = feature.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Feature '{feature_key}' does not provide a usable names list: {names!r}")
    return [str(name) for name in names]


def resolve_named_indices(feature_names: List[str], required_names: List[str], feature_key: str) -> List[int]:
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in required_names if name not in name_to_index]
    if missing:
        raise ValueError(
            f"Feature '{feature_key}' is missing required names {missing}. "
            f"Available names: {feature_names}"
        )
    return [name_to_index[name] for name in required_names]


def build_source_layout(src_info: Dict[str, Any]) -> Dict[str, List[int]]:
    obs_names = get_feature_names(src_info, "observation.state")
    act_names = get_feature_names(src_info, "actions")

    return {
        "obs_left_tcp": resolve_named_indices(obs_names, LEFT_TCP_NAMES, "observation.state"),
        "obs_right_tcp": resolve_named_indices(obs_names, RIGHT_TCP_NAMES, "observation.state"),
        "act_left_tcp": resolve_named_indices(act_names, LEFT_TCP_NAMES, "actions"),
        "act_right_tcp": resolve_named_indices(act_names, RIGHT_TCP_NAMES, "actions"),
        "obs_left_pinch": resolve_named_indices(obs_names, LEFT_PINCH_NAMES, "observation.state"),
        "obs_right_pinch": resolve_named_indices(obs_names, RIGHT_PINCH_NAMES, "observation.state"),
        "act_left_pinch": resolve_named_indices(act_names, LEFT_PINCH_NAMES, "actions"),
        "act_right_pinch": resolve_named_indices(act_names, RIGHT_PINCH_NAMES, "actions"),
        "obs_left_pressure": resolve_named_indices(obs_names, LEFT_FINGER_PRESSURE_NAMES, "observation.state"),
        "obs_right_pressure": resolve_named_indices(obs_names, RIGHT_FINGER_PRESSURE_NAMES, "observation.state"),
        "obs_left_force": resolve_named_indices(obs_names, LEFT_WRIST_FORCE_NAMES, "observation.state"),
        "obs_right_force": resolve_named_indices(obs_names, RIGHT_WRIST_FORCE_NAMES, "observation.state"),
    }


def extract_columns_by_index(arr: np.ndarray, indices: List[int], row_count: int) -> np.ndarray:
    out = np.zeros((row_count, len(indices)), dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2:
        return out
    out[:, :] = arr[:, indices]
    return out


def convert_episode_table(src_parquet: Path, source_layout: Dict[str, List[int]]) -> Tuple[pa.Table, Dict[str, np.ndarray]]:
    table = pq.read_table(src_parquet)
    row_count = table.num_rows

    obs = column_to_2d_float(table, "observation.state", row_count)
    act = column_to_2d_float(table, "actions", row_count)

    left_tcp_obs = extract_columns_by_index(obs, source_layout["obs_left_tcp"], row_count)
    right_tcp_obs = extract_columns_by_index(obs, source_layout["obs_right_tcp"], row_count)
    left_tcp_act = extract_columns_by_index(act, source_layout["act_left_tcp"], row_count)
    right_tcp_act = extract_columns_by_index(act, source_layout["act_right_tcp"], row_count)

    left_delta = compute_delta_tcp(left_tcp_obs, left_tcp_act).astype(np.float32)
    right_delta = compute_delta_tcp(right_tcp_obs, right_tcp_act).astype(np.float32)

    converted: Dict[str, np.ndarray] = {
        "observation.state.left_tcp": left_tcp_obs.astype(np.float32),
        "observation.state.right_tcp": right_tcp_obs.astype(np.float32),
        "observation.state.left_delta_tcp": left_delta,
        "observation.state.right_delta_tcp": right_delta,
        "observation.state.left_pinch": mean_indices_with_zeros(obs, source_layout["obs_left_pinch"], row_count),
        "observation.state.right_pinch": mean_indices_with_zeros(obs, source_layout["obs_right_pinch"], row_count),
        "observation.state.left_finger_pressure": extract_columns_by_index(obs, source_layout["obs_left_pressure"], row_count),
        "observation.state.right_finger_pressure": extract_columns_by_index(obs, source_layout["obs_right_pressure"], row_count),
        "observation.state.left_wrist_force": extract_columns_by_index(obs, source_layout["obs_left_force"], row_count),
        "observation.state.right_wrist_force": extract_columns_by_index(obs, source_layout["obs_right_force"], row_count),
        "action.left_delta_tcp": left_delta.astype(np.float32),
        "action.right_delta_tcp": right_delta.astype(np.float32),
        "action.left_pinch": mean_indices_with_zeros(act, source_layout["act_left_pinch"], row_count),
        "action.right_pinch": mean_indices_with_zeros(act, source_layout["act_right_pinch"], row_count),
        "task.progress.chest": np.zeros(row_count, dtype=np.float32),
        "task.progress.left": np.zeros(row_count, dtype=np.float32),
        "task.progress.right": np.zeros(row_count, dtype=np.float32),
        "teleoperated": get_scalar_column_or_default(table, "teleoperated", row_count, np.int64, 0),
        "timestamp": get_scalar_column_or_default(table, "timestamp", row_count, np.float32, 0.0),
        "frame_index": get_scalar_column_or_default(table, "frame_index", row_count, np.int64, 0),
        "episode_index": get_scalar_column_or_default(table, "episode_index", row_count, np.int64, 0),
        "index": get_scalar_column_or_default(table, "index", row_count, np.int64, 0),
        "task_index": get_scalar_column_or_default(table, "task_index", row_count, np.int64, 0),
    }

    arrow_cols: Dict[str, pa.Array] = {}
    for name in CONVERTED_COLUMN_ORDER:
        arr = converted[name]
        if name in VECTOR_DIMS:
            arrow_cols[name] = to_fixed_size_list(arr, pa.float32())
        elif name in SCALAR_FLOAT_FEATURES:
            arrow_cols[name] = pa.array(arr, type=pa.float32())
        elif name in SCALAR_INT_FEATURES:
            arrow_cols[name] = pa.array(arr, type=pa.int64())
        else:
            raise KeyError(f"Unhandled converted column: {name}")

    return pa.table(arrow_cols), converted


def find_source_video(src_root: Path, src_key: str, episode_number: int) -> Path:
    base = src_root / "videos" / "chunk-000" / src_key / f"episode_{episode_number:06d}"
    for ext in VIDEO_SOURCE_EXTS:
        candidate = Path(f"{base}{ext}")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing source video for {src_key}, episode {episode_number:06d}. "
        f"Tried extensions: {', '.join(VIDEO_SOURCE_EXTS)}"
    )


def transcode_video_resize_to_mp4(
    src_video: Path,
    dst_video: Path,
    width: int,
    height: int,
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    if dst_video.exists():
        dst_video.unlink()

    # Force 224x224 RGB-like 3-channel output via yuv420p mp4.
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed for {src_video} -> {dst_video}\n{stderr}")
# def transcode_video_resize_to_mp4(
#     src_video: Path,
#     dst_video: Path,
#     width: int,
#     height: int,
# ) -> None:
#     import cv2

#     dst_video.parent.mkdir(parents=True, exist_ok=True)

#     if dst_video.exists():
#         dst_video.unlink()

#     cap = cv2.VideoCapture(str(src_video))

#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open source video: {src_video}")

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 1e-3:
#         fps = 30.0

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(str(dst_video), fourcc, fps, (width, height))

#     if not out.isOpened():
#         raise RuntimeError(f"Cannot create output video: {dst_video}")

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame = cv2.resize(frame, (width, height))
#             out.write(frame)

#     finally:
#         cap.release()
#         out.release()

def file_size_mb(paths: Iterable[Path]) -> int:
    total = sum(p.stat().st_size for p in paths if p.is_file())
    return int(round(total / (1024.0 * 1024.0)))


def build_tasks_table(
    tasks_jsonl: List[Dict[str, Any]],
    task_text: str,
) -> Tuple[pa.Table, Dict[int, str]]:
    if tasks_jsonl:
        rows = sorted(tasks_jsonl, key=lambda x: int(x.get("task_index", 0)))
        task_index = [int(r.get("task_index", i)) for i, r in enumerate(rows)]
        task_name = [task_text for _ in rows]
    else:
        task_index = [0]
        task_name = [task_text]
    task_map = {idx: name for idx, name in zip(task_index, task_name)}
    table = pa.table({"task_index": pa.array(task_index, type=pa.int64()), "task": pa.array(task_name)})
    return table, task_map


def build_info_json(
    src_info: Dict[str, Any],
    video_map: List[Tuple[str, str]],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    chunk_size: int,
    data_size_mb: int,
    video_size_mb: int,
    video_width: int,
    video_height: int,
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}

    src_features = src_info.get("features", {})
    for src_key, dst_key in video_map:
        src_feature = src_features.get(src_key, {})
        video_info = dict(src_feature.get("info", {}))
        video_info["video.channels"] = 3
        video_info["video.height"] = int(video_height)
        video_info["video.width"] = int(video_width)
        video_info["video.codec"] = "h264"
        video_info["video.pix_fmt"] = "yuv420p"
        video_info["video.is_depth_map"] = False
        video_info["has_audio"] = False
        if "video.fps" not in video_info:
            video_info["video.fps"] = src_info.get("fps", 30)
        features[dst_key] = {
            "dtype": "video",
            "shape": [3, int(video_height), int(video_width)],
            "names": ["channels", "height", "width"],
            "info": video_info,
        }

    for name, dim in VECTOR_DIMS.items():
        features[name] = {"dtype": "float32", "shape": [dim], "names": None}
    for name in SCALAR_FLOAT_FEATURES:
        features[name] = {"dtype": "float32", "shape": [1], "names": None}
    for name in SCALAR_INT_FEATURES:
        features[name] = {"dtype": "int64", "shape": [1], "names": None}

    return {
        "codebase_version": "v3.0",
        "robot_type": src_info.get("robot_type"),
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "chunks_size": int(chunk_size),
        "data_files_size_in_mb": int(data_size_mb),
        "video_files_size_in_mb": int(video_size_mb),
        "fps": int(src_info.get("fps", 30)),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")

    ensure_clean_dir(dst, args.overwrite)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    src_info = read_json(src / "meta" / "info.json")
    video_map = parse_video_map(args.video_map) if args.video_map else resolve_default_video_map(src_info)
    source_layout = build_source_layout(src_info)
    tasks_jsonl = read_jsonl(src / "meta" / "tasks.jsonl")
    episodes_jsonl = read_jsonl(src / "meta" / "episodes.jsonl")

    episode_files = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet found under {src / 'data' / 'chunk-000'}")

    tasks_table, task_map = build_tasks_table(tasks_jsonl, args.task_text)
    tasks_out = dst / "meta" / "tasks.parquet"
    pq.write_table(tasks_table, tasks_out)

    episode_task_name: Dict[int, str] = {}
    for row in episodes_jsonl:
        ep_idx = int(row.get("episode_index", -1))
        task_field = row.get("tasks", "")
        if isinstance(task_field, list):
            task_name = str(task_field[0]) if task_field else ""
        else:
            task_name = str(task_field)
        if ep_idx >= 0:
            episode_task_name[ep_idx] = task_name

    numeric_features = list(VECTOR_DIMS.keys()) + SCALAR_FLOAT_FEATURES + SCALAR_INT_FEATURES
    default_task_index = int(tasks_table.column("task_index")[0].as_py()) if tasks_table.num_rows > 0 else 0
    total_frames = 0
    global_feature_values: Dict[str, List[np.ndarray]] = {k: [] for k in numeric_features}

    for file_index, episode_file in enumerate(episode_files):
        ep_table, ep_np = convert_episode_table(episode_file, source_layout)
        row_count = ep_table.num_rows
        total_frames += row_count

        data_out = dst / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        pq.write_table(ep_table, data_out)

        for feature in numeric_features:
            global_feature_values[feature].append(np.asarray(ep_np[feature], dtype=np.float64))

        episode_index = int(ep_np["episode_index"][0]) if row_count > 0 else file_index
        idx_start = int(ep_np["index"][0]) if row_count > 0 else 0
        idx_end = int(ep_np["index"][-1]) + 1 if row_count > 0 else 0
        ts_start = float(ep_np["timestamp"][0]) if row_count > 0 else 0.0
        ts_end = float(ep_np["timestamp"][-1]) if row_count > 0 else 0.0
        task_idx = int(ep_np["task_index"][0]) if row_count > 0 else default_task_index

        task_name = args.task_text
        if not task_name:
            task_name = episode_task_name.get(episode_index)
            if task_name is None:
                task_name = task_map.get(task_idx, "")

        episode_number_str = episode_file.stem.split("_")[-1]
        source_episode_number = int(episode_number_str)

        ep_row: Dict[str, List[Any]] = {
            "episode_index": [episode_index],
            "tasks": [[task_name] if task_name else []],
            "length": [int(row_count)],
            "data/chunk_index": [0],
            "data/file_index": [file_index],
            "dataset_from_index": [idx_start],
            "dataset_to_index": [idx_end],
            "meta/episodes/chunk_index": [0],
            "meta/episodes/file_index": [file_index],
        }

        for src_key, dst_key in video_map:
            src_video = find_source_video(src, src_key, source_episode_number)
            dst_video = dst / "videos" / dst_key / "chunk-000" / f"file-{file_index:03d}.mp4"
            transcode_video_resize_to_mp4(
                src_video=src_video,
                dst_video=dst_video,
                width=args.video_width,
                height=args.video_height,
            )

            ep_row[f"videos/{dst_key}/chunk_index"] = [0]
            ep_row[f"videos/{dst_key}/file_index"] = [file_index]
            ep_row[f"videos/{dst_key}/from_timestamp"] = [ts_start]
            ep_row[f"videos/{dst_key}/to_timestamp"] = [ts_end]

        for feature in numeric_features:
            feature_stats = summarize_array(ep_np[feature])
            for stat_key in STATS_KEYS:
                ep_row[f"stats/{feature}/{stat_key}"] = [feature_stats[stat_key]]

        for src_key, dst_key in video_map:
            _ = src_key
            video_stats = summarize_video_placeholder(row_count, 3)
            for stat_key in STATS_KEYS:
                ep_row[f"stats/{dst_key}/{stat_key}"] = [video_stats[stat_key]]

        episodes_columns: Dict[str, pa.Array] = {
            "episode_index": pa.array(ep_row["episode_index"], type=pa.int64()),
            "tasks": pa.array(ep_row["tasks"], type=pa.list_(pa.string())),
            "length": pa.array(ep_row["length"], type=pa.int64()),
            "data/chunk_index": pa.array(ep_row["data/chunk_index"], type=pa.int64()),
            "data/file_index": pa.array(ep_row["data/file_index"], type=pa.int64()),
            "dataset_from_index": pa.array(ep_row["dataset_from_index"], type=pa.int64()),
            "dataset_to_index": pa.array(ep_row["dataset_to_index"], type=pa.int64()),
            "meta/episodes/chunk_index": pa.array(ep_row["meta/episodes/chunk_index"], type=pa.int64()),
            "meta/episodes/file_index": pa.array(ep_row["meta/episodes/file_index"], type=pa.int64()),
        }

        for _, dst_key in video_map:
            episodes_columns[f"videos/{dst_key}/chunk_index"] = pa.array(
                ep_row[f"videos/{dst_key}/chunk_index"], type=pa.int64()
            )
            episodes_columns[f"videos/{dst_key}/file_index"] = pa.array(
                ep_row[f"videos/{dst_key}/file_index"], type=pa.int64()
            )
            episodes_columns[f"videos/{dst_key}/from_timestamp"] = pa.array(
                ep_row[f"videos/{dst_key}/from_timestamp"], type=pa.float64()
            )
            episodes_columns[f"videos/{dst_key}/to_timestamp"] = pa.array(
                ep_row[f"videos/{dst_key}/to_timestamp"], type=pa.float64()
            )

        for feature in numeric_features:
            for stat_key in STATS_KEYS:
                episodes_columns[f"stats/{feature}/{stat_key}"] = pa.array(
                    ep_row[f"stats/{feature}/{stat_key}"]
                )

        for _, dst_key in video_map:
            for stat_key in STATS_KEYS:
                episodes_columns[f"stats/{dst_key}/{stat_key}"] = pa.array(
                    ep_row[f"stats/{dst_key}/{stat_key}"]
                )

        episodes_out = dst / "meta" / "episodes" / "chunk-000" / f"file-{file_index:03d}.parquet"
        pq.write_table(pa.table(episodes_columns), episodes_out)

    stats_json: Dict[str, Any] = {}
    for feature in numeric_features:
        values = np.concatenate(global_feature_values[feature], axis=0)
        stats_json[feature] = summarize_array(values)

    for src_key, dst_key in video_map:
        _ = src_key
        stats_json[dst_key] = summarize_video_placeholder(total_frames, 3)

    with (dst / "meta" / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=False)

    data_size_mb = file_size_mb((dst / "data").rglob("*"))
    video_size_mb = file_size_mb((dst / "videos").rglob("*"))
    info_json = build_info_json(
        src_info=src_info,
        video_map=video_map,
        total_episodes=len(episode_files),
        total_frames=total_frames,
        total_tasks=tasks_table.num_rows,
        chunk_size=args.chunk_size,
        data_size_mb=data_size_mb,
        video_size_mb=video_size_mb,
        video_width=args.video_width,
        video_height=args.video_height,
    )
    with (dst / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info_json, f, indent=2, ensure_ascii=False)

    print("Conversion complete.")
    print(f"  source: {src}")
    print(f"  target: {dst}")
    print(f"  episodes: {len(episode_files)}")
    print(f"  frames: {total_frames}")
    print("  files:")
    print(f"    - {dst / 'data' / 'chunk-000' / 'file-000.parquet'} ..")
    print(f"    - {tasks_out}")
    print(f"    - {dst / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'} ..")
    print(f"    - {dst / 'meta' / 'info.json'}")
    print(f"    - {dst / 'meta' / 'stats.json'}")


if __name__ == "__main__":
    main()
