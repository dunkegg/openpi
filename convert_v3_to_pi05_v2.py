#!/usr/bin/env python3
"""
Convert a LeRobot V3 dataset with sub-keyed state/action features into a flat
PI0.5-compatible LeRobot **V2.1** dataset.

OpenPI's pinned lerobot expects V2.1 format:
  - meta/info.json        (codebase_version "v2.1")
  - meta/tasks.jsonl
  - meta/episodes.jsonl
  - meta/episodes_stats.jsonl
  - meta/stats.json       (global aggregated stats)
  - Per-episode parquet:   data/chunk-{chunk:03d}/episode_{idx:06d}.parquet
  - Per-episode videos:    videos/chunk-{chunk:03d}/{video_key}/episode_{idx:06d}.mp4
  - meta/pi05_manifest.json  (our custom manifest for downstream use)

The conversion:
  1. Reads V3 source dataset (consolidated parquet + concatenated videos)
  2. Concatenates observation.state.* sub-keys -> observation.state (single flat tensor)
  3. Concatenates action.* sub-keys            -> action            (single flat tensor)
  4. Splits consolidated data parquet into per-episode parquet files
  5. Splits concatenated videos into per-episode videos via ffmpeg
  6. Converts metadata to V2.1 JSONL format
  7. Generates meta/pi05_manifest.json

Usage:
    python scripts/convert_dataset_for_pi05_v2.py \\
        --src datasets/jean_auto_delta_merged \\
        --dst datasets/jean_auto_delta_merged_pi05

    # With feature selection (optional):
    python scripts/convert_dataset_for_pi05_v2.py \\
        --src datasets/jean_auto_delta_merged \\
        --dst datasets/jean_auto_delta_merged_pi05 \\
        --state-keys observation.state.right_tcp,observation.state.right_pinch \\
        --action-keys action.right_delta_tcp,action.right_pinch
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow.parquet as pq
import pandas as pd


CHUNKS_SIZE = 1000  # default LeRobot chunk size

# V2.1 path templates
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# V3 path templates
V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

MIN_VIDEO_DURATION = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sub_keys(features: dict, prefix: str) -> list[str]:
    """Return all feature keys that start with `prefix.`, sorted alphabetically."""
    return sorted(k for k in features if k.startswith(prefix + "."))


def _filter_keys(all_keys: list[str], whitelist: list[str] | None) -> list[str]:
    if whitelist is None:
        return all_keys
    whitelist_set = set(whitelist)
    filtered = [k for k in all_keys if k in whitelist_set]
    missing = whitelist_set - set(filtered)
    if missing:
        raise ValueError(f"Whitelist keys not found in dataset: {missing}")
    return filtered


def _concat_row_vectors(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Concatenate per-key values into a single flat vector per row. Returns (N, total_dim)."""
    rows = []
    for _, row in df.iterrows():
        parts = []
        for k in keys:
            v = row[k]
            if np.isscalar(v):
                parts.append(np.array([v], dtype=np.float32))
            else:
                parts.append(np.asarray(v, dtype=np.float32).ravel())
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def _total_dim(features: dict, keys: list[str]) -> int:
    total = 0
    for k in keys:
        shape = features[k]["shape"]
        total += int(np.prod(shape))
    return total


def _short_key(full_key: str, prefix: str) -> str:
    return full_key[len(prefix) + 1:]


def _build_manifest(features: dict, state_keys: list[str], action_keys: list[str],
                     state_prefix: str, action_prefix: str, fps: int) -> dict:
    state_entries = []
    offset = 0
    for k in state_keys:
        dim = int(np.prod(features[k]["shape"]))
        state_entries.append({
            "original_key": k,
            "short_key": _short_key(k, state_prefix),
            "dim": dim,
            "start": offset,
            "end": offset + dim,
        })
        offset += dim

    action_entries = []
    offset = 0
    for k in action_keys:
        dim = int(np.prod(features[k]["shape"]))
        action_entries.append({
            "original_key": k,
            "short_key": _short_key(k, action_prefix),
            "dim": dim,
            "start": offset,
            "end": offset + dim,
        })
        offset += dim

    image_keys = {}
    for k, v in features.items():
        if v.get("dtype") == "video" or k.startswith("observation.images."):
            short = k.split(".")[-1]
            image_keys[short] = k

    return {
        "state_keys": state_entries,
        "action_keys": action_entries,
        "image_keys": image_keys,
        "state_dim": sum(e["dim"] for e in state_entries),
        "action_dim": sum(e["dim"] for e in action_entries),
        "fps": fps,
    }


def _to_serializable(value):
    """Convert numpy/pyarrow values to JSON-safe Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def _unflatten_dict(flat: dict, sep: str = "/") -> dict:
    """Unflatten a dict with keys like 'a/b/c' into nested dicts."""
    result = {}
    for key, value in flat.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def _extract_video_segment(src: Path, dst: Path, start: float, end: float, fps: int) -> None:
    """Extract a video segment with timestamp reset so the first frame starts at 0.0s."""
    frame_dt = 1.0 / float(max(fps, 1))
    duration = max((end - start) + frame_dt, MIN_VIDEO_DURATION)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"trim=start={start:.6f}:duration={duration:.6f},setpts=PTS-STARTPTS,fps={fps}",
        "-vsync", "cfr",
        "-an",
        "-c:v", "libx264",
        "-g", "1",
        "-keyint_min", "1",
        "-bf", "0",
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        "-y", str(dst),
    ]
    subprocess.run(cmd, check=True, timeout=300, capture_output=True)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _concat_stats(stats: dict, keys: list[str]) -> dict:
    """Concatenate per-element stats for sub-keys into one stat block."""
    all_stat_names = set()
    for k in keys:
        if k in stats:
            all_stat_names.update(stats[k].keys())

    merged = {}
    for stat_name in sorted(all_stat_names):
        if stat_name == "count":
            merged["count"] = stats[keys[0]]["count"]
            continue
        combined = []
        for k in keys:
            v = stats.get(k, {}).get(stat_name, [])
            if isinstance(v, (int, float)):
                v = [v]
            combined.extend(v)
        merged[stat_name] = combined
    return merged


def _compute_episode_stats_for_column(values: np.ndarray) -> dict:
    """Compute per-column stats matching V2.1 episodes_stats format."""
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [len(values)],
        "q01": np.percentile(values, 1, axis=0).tolist(),
        "q10": np.percentile(values, 10, axis=0).tolist(),
        "q50": np.percentile(values, 50, axis=0).tolist(),
        "q90": np.percentile(values, 90, axis=0).tolist(),
        "q99": np.percentile(values, 99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_dataset(src: Path, dst: Path,
                    state_whitelist: list[str] | None = None,
                    action_whitelist: list[str] | None = None,
                    default_task: str | None = None) -> None:
    src = Path(src)
    dst = Path(dst)

    if dst.exists():
        print(f"Destination {dst} already exists - removing it first.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # ------------------------------------------------------------------
    # Load V3 metadata
    # ------------------------------------------------------------------
    with open(src / "meta" / "info.json") as f:
        info = json.load(f)

    features = info["features"]
    fps = info.get("fps", 30)
    chunks_size = info.get("chunks_size", CHUNKS_SIZE)
    total_episodes = info.get("total_episodes", 0)

    # Load V3 episode records
    episodes_dir = src / "meta" / "episodes"
    ep_pq_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not ep_pq_files:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}")

    episode_records = []
    for pq_path in ep_pq_files:
        table = pq.read_table(pq_path)
        episode_records.extend(table.to_pylist())
    episode_records.sort(key=lambda r: int(r["episode_index"]))

    # Load V3 tasks
    tasks_pq = src / "meta" / "tasks.parquet"
    tasks_list = pq.read_table(tasks_pq).to_pylist() if tasks_pq.exists() else []

    # Identify video keys
    video_keys = [k for k, ft in features.items() if ft.get("dtype") == "video"]

    # ------------------------------------------------------------------
    # Determine state/action keys and dims
    # ------------------------------------------------------------------
    state_keys = _filter_keys(_get_sub_keys(features, "observation.state"), state_whitelist)
    action_keys = _filter_keys(_get_sub_keys(features, "action"), action_whitelist)

    state_dim = _total_dim(features, state_keys)
    action_dim = _total_dim(features, action_keys)

    print(f"State sub-keys ({len(state_keys)}): {state_keys}")
    print(f"  -> observation.state dim = {state_dim}")
    print(f"Action sub-keys ({len(action_keys)}): {action_keys}")
    print(f"  -> action dim = {action_dim}")

    # Build manifest
    manifest = _build_manifest(features, state_keys, action_keys,
                                "observation.state", "action", fps)

    # ------------------------------------------------------------------
    # Build new V2.1 features dict
    # ------------------------------------------------------------------
    new_features = {}
    for k, v in features.items():
        if k in state_keys or k in action_keys:
            continue
        new_features[k] = v

    new_features["observation.state"] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": None,
    }
    new_features["action"] = {
        "dtype": "float32",
        "shape": [action_dim],
        "names": None,
    }

    # ------------------------------------------------------------------
    # Write V2.1 info.json
    # ------------------------------------------------------------------
    new_info = dict(info)
    new_info["codebase_version"] = "v2.1"
    new_info["features"] = new_features
    new_info["data_path"] = V21_DATA_PATH
    if video_keys:
        new_info["video_path"] = V21_VIDEO_PATH
    else:
        new_info["video_path"] = None
    # Remove V3-only fields
    new_info.pop("data_files_size_in_mb", None)
    new_info.pop("video_files_size_in_mb", None)
    # Add V2.1 fields
    new_info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    new_info["total_videos"] = total_episodes * len(video_keys)

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)
    print(f"Wrote {dst / 'meta' / 'info.json'}")

    # ------------------------------------------------------------------
    # Write V2.1 tasks.jsonl
    # ------------------------------------------------------------------
    tasks_path = dst / "meta" / "tasks.jsonl"
    with jsonlines.open(tasks_path, mode="w") as writer:
        for t in sorted(tasks_list, key=lambda x: x["task_index"]):
            task_str = t["task"]
            if not task_str and default_task:
                task_str = default_task
            if not task_str:
                raise ValueError(
                    f"Task {t['task_index']} has an empty description. "
                    "OpenPI requires a non-empty prompt for training. "
                    "Use --default-task to provide one (e.g. --default-task 'pick up the cloth')."
                )
            writer.write({"task_index": int(t["task_index"]), "task": task_str})
    print(f"Wrote {tasks_path}")

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    with open(dst / "meta" / "pi05_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {dst / 'meta' / 'pi05_manifest.json'}")

    # ------------------------------------------------------------------
    # Load all V3 data and convert parquet: consolidated -> per-episode
    # ------------------------------------------------------------------
    # Group episode records by their source data file
    episodes_by_file: dict[tuple[int, int], list[dict]] = {}
    for rec in episode_records:
        key = (int(rec["data/chunk_index"]), int(rec["data/file_index"]))
        episodes_by_file.setdefault(key, []).append(rec)

    print(f"\nConverting data to per-episode parquet files...")

    # We'll also collect per-episode stats for the new merged columns
    all_episode_stats: dict[int, dict] = {}

    for (chunk_idx, file_idx), records in sorted(episodes_by_file.items()):
        src_pq_path = src / V3_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not src_pq_path.exists():
            raise FileNotFoundError(f"Expected source parquet: {src_pq_path}")

        df_full = pd.read_parquet(src_pq_path)
        records = sorted(records, key=lambda r: int(r["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for rec in records:
            ep_idx = int(rec["episode_index"])
            start = int(rec["dataset_from_index"]) - file_offset
            stop = int(rec["dataset_to_index"]) - file_offset

            df_ep = df_full.iloc[start:stop].copy()

            # Concatenate state and action sub-keys
            state_matrix = _concat_row_vectors(df_ep, state_keys)
            action_matrix = _concat_row_vectors(df_ep, action_keys)

            # Drop old sub-key columns
            df_ep = df_ep.drop(columns=state_keys + action_keys)

            # Insert merged columns
            df_ep["observation.state"] = list(state_matrix)
            df_ep["action"] = list(action_matrix)

            # Write per-episode parquet
            ep_chunk = ep_idx // chunks_size
            dst_pq = dst / V21_DATA_PATH.format(episode_chunk=ep_chunk, episode_index=ep_idx)
            dst_pq.parent.mkdir(parents=True, exist_ok=True)
            df_ep.to_parquet(dst_pq, index=False)

            # Compute per-episode stats for the merged columns
            ep_stats = {}
            ep_stats["observation.state"] = _compute_episode_stats_for_column(state_matrix)
            ep_stats["action"] = _compute_episode_stats_for_column(action_matrix)
            all_episode_stats[ep_idx] = ep_stats

            print(f"  episode_{ep_idx:06d}.parquet  ({len(df_ep)} frames)")

    # ------------------------------------------------------------------
    # Convert videos: concatenated -> per-episode via ffmpeg
    # ------------------------------------------------------------------
    if video_keys:
        print(f"\nSplitting videos into per-episode files...")
        for vk in video_keys:
            for rec in episode_records:
                ep_idx = int(rec["episode_index"])
                chunk_idx = int(rec[f"videos/{vk}/chunk_index"])
                file_idx = int(rec[f"videos/{vk}/file_index"])
                from_ts = float(rec[f"videos/{vk}/from_timestamp"])
                to_ts = float(rec[f"videos/{vk}/to_timestamp"])

                src_video = src / V3_VIDEO_PATH.format(
                    video_key=vk, chunk_index=chunk_idx, file_index=file_idx)
                ep_chunk = ep_idx // chunks_size
                dst_video = dst / V21_VIDEO_PATH.format(
                    episode_chunk=ep_chunk, video_key=vk, episode_index=ep_idx)

                _extract_video_segment(src_video, dst_video, from_ts, to_ts, fps=fps)
                print(f"  {vk} episode_{ep_idx:06d}.mp4  [{from_ts:.2f}s - {to_ts:.2f}s]")

    # ------------------------------------------------------------------
    # Write V2.1 episodes.jsonl and episodes_stats.jsonl
    # ------------------------------------------------------------------
    episodes_jsonl_path = dst / "meta" / "episodes.jsonl"
    stats_jsonl_path = dst / "meta" / "episodes_stats.jsonl"

    with jsonlines.open(episodes_jsonl_path, mode="w") as ep_writer, \
         jsonlines.open(stats_jsonl_path, mode="w") as stats_writer:

        for rec in episode_records:
            ep_idx = int(rec["episode_index"])

            # Episode metadata: exclude V3-specific fields
            legacy_ep = {}
            for key, value in rec.items():
                if (key.startswith("data/") or key.startswith("videos/") or
                    key.startswith("stats/") or key.startswith("meta/") or
                    key in ("dataset_from_index", "dataset_to_index")):
                    continue
                legacy_ep[key] = _to_serializable(value)

            if "length" not in legacy_ep:
                if "dataset_from_index" in rec and "dataset_to_index" in rec:
                    legacy_ep["length"] = int(rec["dataset_to_index"]) - int(rec["dataset_from_index"])

            ep_writer.write(legacy_ep)

            # Episode stats: merge V3 per-feature stats + our new merged column stats
            stats_flat = {k: rec[k] for k in rec if k.startswith("stats/")}
            stats_nested = _unflatten_dict(stats_flat).get("stats", {})

            # Remove stats for old sub-keys, add stats for merged columns
            for sk in state_keys:
                flat_sk = sk  # e.g. "observation.state.right_tcp"
                # V3 stats keys use the feature name directly
                stats_nested.pop(flat_sk, None)
            for ak in action_keys:
                stats_nested.pop(ak, None)

            # Add our computed episode stats for merged columns
            if ep_idx in all_episode_stats:
                stats_nested["observation.state"] = all_episode_stats[ep_idx]["observation.state"]
                stats_nested["action"] = all_episode_stats[ep_idx]["action"]

            stats_writer.write({
                "episode_index": ep_idx,
                "stats": _to_serializable(stats_nested),
            })

    print(f"Wrote {episodes_jsonl_path}")
    print(f"Wrote {stats_jsonl_path}")

    # ------------------------------------------------------------------
    # Write V2.1 global stats.json (if source has one, update it)
    # ------------------------------------------------------------------
    src_stats_path = src / "meta" / "stats.json"
    if src_stats_path.exists():
        with open(src_stats_path) as f:
            global_stats = json.load(f)

        new_global_stats = {}
        for k, v in global_stats.items():
            if k in state_keys or k in action_keys:
                continue
            new_global_stats[k] = v

        new_global_stats["observation.state"] = _concat_stats(global_stats, state_keys)
        new_global_stats["action"] = _concat_stats(global_stats, action_keys)

        with open(dst / "meta" / "stats.json", "w") as f:
            json.dump(_to_serializable(new_global_stats), f, indent=2)
        print(f"Wrote {dst / 'meta' / 'stats.json'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nDone. V2.1 dataset written to: {dst}")
    print(f"  codebase_version      : v2.1")
    print(f"  observation.state dim : {state_dim}")
    print(f"  action dim            : {action_dim}")
    print(f"  episodes              : {total_episodes}")
    print(f"  video keys            : {video_keys}")
    print(f"\nState key order (must match inference preprocessing):")
    for entry in manifest["state_keys"]:
        print(f"  [{entry['start']:2d}:{entry['end']:2d}] {entry['original_key']}  (dim={entry['dim']})")
    print(f"\nAction key order:")
    for entry in manifest["action_keys"]:
        print(f"  [{entry['start']:2d}:{entry['end']:2d}] {entry['original_key']}  (dim={entry['dim']})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="Path to the source V3 dataset directory")
    parser.add_argument("--dst", required=True, help="Path to write the V2.1 converted dataset")
    parser.add_argument(
        "--state-keys", default=None,
        help="Comma-separated whitelist of observation.state.* keys to include (default: all)",
    )
    parser.add_argument(
        "--action-keys", default=None,
        help="Comma-separated whitelist of action.* keys to include (default: all)",
    )
    parser.add_argument(
        "--default-task", default=None,
        help="Default task description to use when source tasks are empty (e.g. 'pick up the cloth')",
    )
    args = parser.parse_args()

    state_whitelist = args.state_keys.split(",") if args.state_keys else None
    action_whitelist = args.action_keys.split(",") if args.action_keys else None

    convert_dataset(Path(args.src), Path(args.dst), state_whitelist, action_whitelist, args.default_task)


if __name__ == "__main__":
    main()
