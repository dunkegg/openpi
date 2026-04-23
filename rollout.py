#!/usr/bin/env python3
"""
模型 Rollout 脚本

用于在训练完成后对模型进行评估测试，支持以下模式：
1. 从保存的checkpoint加载模型并运行rollout
2. 支持保存动作预测结果和可视化

使用方法:
    python rollout.py --config zj_humanoid_pi0_test --checkpoint_dir /path/to/checkpoint --output_dir ./rollout_results
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch

from openpi.training import config as _config
from openpi.policies import policy_config


def parse_args():
    parser = argparse.ArgumentParser(description="模型 Rollout 评估脚本")
    
    # 模型配置
    parser.add_argument("--config", type=str, default="pi05_wh1",
                        help="训练配置名称")
    parser.add_argument("--checkpoint_dir", type=str, 
                        default="checkpoints/pi05_wh1/test_express_1/80000",
                        help="模型checkpoint路径")
    parser.add_argument("--step", type=int, default=None,
                        help="指定特定step的checkpoint，若不指定则使用checkpoint_dir")
    
    # 数据配置
    parser.add_argument("--data_dir", type=str,
                        default="data/lerobot/express_v2pi",
                        help="数据目录路径")
    parser.add_argument("--num_episodes", type=int, default=30,
                        help="评估的episode数量")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="起始episode编号")
    
    # 推理配置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="扩散模型推理步数")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./rollout_results2",
                        help="结果输出目录")
    parser.add_argument("--save_actions", action="store_true", default=True,
                        help="是否保存预测动作")
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="是否保存可视化图像")
    parser.add_argument("--save_video", action="store_true", default=False,
                        help="是否保存rollout视频")
    
    return parser.parse_args()


def load_episode_video(data_dir: Path, episode_idx: int) -> cv2.VideoCapture:
    """加载指定episode的mp4视频"""
    chest_video_path = data_dir / "videos" / "chunk-000" / "observation.images.chest" / f"episode_{episode_idx:06d}.mp4"
    left_video_path = data_dir / "videos" / "chunk-000" / "observation.images.chest" / f"episode_{episode_idx:06d}.mp4"
    right_video_path = data_dir / "videos" / "chunk-000" / "observation.images.chest" / f"episode_{episode_idx:06d}.mp4"
    
    if not chest_video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {chest_video_path}")
    
    cap_chest = cv2.VideoCapture(str(chest_video_path))
    cap_left = cv2.VideoCapture(str(left_video_path))
    cap_right = cv2.VideoCapture(str(right_video_path))
    if not cap_chest.isOpened():
        raise RuntimeError(f"无法打开视频: {chest_video_path}")
    
    return [cap_chest, cap_right, cap_left]


def load_episode_data(data_dir: Path, episode_idx: int) -> tuple:
    """加载指定episode的数据，返回(parquet数据, 视频捕获器)"""
    chunk_idx = episode_idx // 1000   # 假设每1000个episode一个chunk
    episode_file = data_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    
    if not episode_file.exists():
        raise FileNotFoundError(f"Episode文件不存在: {episode_file}")
    
    try:
        import pandas as pd
        df = pd.read_parquet(episode_file)
        
        # 加载对应的mp4视频
        caps = load_episode_video(data_dir, episode_idx)
        
        return df, caps
    except Exception as e:
        raise RuntimeError(f"加载episode {episode_idx} 失败: {e}")


def release_video(cap: cv2.VideoCapture):
    """释放视频捕获器"""
    if cap is not None:
        cap.release()


def get_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    """从视频捕获器中读取指定帧"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  #必须用帧索引
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"无法读取视频帧 {frame_idx}")
    # OpenCV读取的是BGR格式，转换为RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def format_observation(
    frame_data: dict, 
    frame_idx: int, 
    caps: list[cv2.VideoCapture],
    prompt: str = "pick tennis"
) -> dict:
    """将原始数据格式化为模型输入格式"""
    chest_image = get_video_frame(caps[0], frame_idx)
    right_image = get_video_frame(caps[1], frame_idx)
    left_image = get_video_frame(caps[2], frame_idx)
    
    return {
        "images" : {
            "cam_low": chest_image,
            "cam_right_wrist": right_image,
            "cam_left_wrist": left_image,
        },
        "state": np.array(frame_data["observation.state"], dtype=np.float32),
        "prompt": prompt,
        "actions": np.array(frame_data["actions"], dtype=np.float32),
    }


# def create_observation_from_demo(policy, obs: dict) -> dict:
#     """创建演示观察（用于首次推理）"""
#     # 复用policy的transforms处理输入
#     demo_obs = {
#         "observation.wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
#         "observation.state": np.zeros(11, dtype=np.float32),
#         "prompt": "pick tennis",
#     }
    
#     # 替换为实际观察
#     demo_obs.update(obs)
#     return demo_obs


def save_visualization(
    output_dir: Path,
    episode_idx: int,
    frames_data: list,
    predicted_actions: list[np.ndarray],
    ground_truth_actions: list[np.ndarray] = None,
    video_cap: cv2.VideoCapture = None,
    action_dim: int = 6,
    fps: int = 10
):
    """保存可视化结果，包含预测动作、真实动作和误差对比"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(frames_data) == 0:
        return
    
    has_ground_truth = ground_truth_actions is not None and len(ground_truth_actions) > 0
    
    if video_cap is not None:
        w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        first_frame = frames_data[0]
        wrist_image = first_frame.get("observation.images.chest")
        if hasattr(wrist_image, 'asnumpy'):
            frame_img = wrist_image.asnumpy()
        elif not isinstance(wrist_image, np.ndarray):
            frame_img = np.array(wrist_image)
        else:
            frame_img = wrist_image.copy()
        if frame_img.dtype != np.uint8:
            frame_img = frame_img.astype(np.uint8)
        if frame_img.shape[0] == 3:
            frame_img = np.transpose(frame_img, (1, 2, 0))
        h, w = frame_img.shape[:2]
    
    # 面板宽度根据是否有真实动作调整
    panel_w = w if not has_ground_truth else int(w * 0.5)
    video_path = output_dir / f"episode_{episode_idx:06d}_rollout.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w + panel_w, h))
    
    for i, (frame, pred_action) in enumerate(zip(frames_data, predicted_actions)):
        if video_cap is not None:
            frame_img = get_video_frame(video_cap, i)
        else:
            wrist_image = frame.get("observation.images.chest")
            if hasattr(wrist_image, 'asnumpy'):
                frame_img = wrist_image.asnumpy()
            elif not isinstance(wrist_image, np.ndarray):
                frame_img = np.array(wrist_image)
            else:
                frame_img = wrist_image.copy()
            if frame_img.dtype != np.uint8:
                frame_img = frame_img.astype(np.uint8)
            if frame_img.shape[0] == 3:
                frame_img = np.transpose(frame_img, (1, 2, 0))
        
        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR) if frame_img.shape[-1] == 3 else frame_img
        
        # 信息面板
        info_panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        y_pos = 30
        
        cv2.putText(info_panel, f"Step: {i}/{len(predicted_actions) - 1}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 40
        
        # 预测动作
        pred_1d = np.asarray(pred_action).flatten()
        cv2.putText(info_panel, f"Predicted ({action_dim}D):", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)
        y_pos += 30
        
        for j in range(min(action_dim, len(pred_1d))):
            val = float(pred_1d[j])
            cv2.putText(info_panel, f"  J{j}: {val:+.4f}", 
                        (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
            y_pos += 22
        
        y_pos += 10
        
        # 真实动作和误差（如果有）
        if has_ground_truth and i < len(ground_truth_actions):
            gt_action = ground_truth_actions[i]
            gt_1d = np.asarray(gt_action).flatten()
            
            cv2.putText(info_panel, f"Ground Truth ({action_dim}D):", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 2)
            y_pos += 30
            
            for j in range(min(action_dim, len(gt_1d))):
                val = float(gt_1d[j])
                cv2.putText(info_panel, f"  J{j}: {val:+.4f}", 
                            (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 180), 1)
                y_pos += 22
            
            y_pos += 10
            
            # 计算并显示误差
            mae = np.mean(np.abs(pred_1d[:action_dim] - gt_1d[:action_dim]))
            mse = np.mean((pred_1d[:action_dim] - gt_1d[:action_dim]) ** 2)
            
            cv2.putText(info_panel, f"Error Metrics:", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 2)
            y_pos += 30
            cv2.putText(info_panel, f"  MAE: {mae:.4f}", 
                        (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 150), 1)
            y_pos += 22
            cv2.putText(info_panel, f"  MSE: {mse:.6f}", 
                        (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 150), 1)
            
            # 每关节误差
            joint_errors = np.abs(pred_1d[:action_dim] - gt_1d[:action_dim])
            y_pos += 30
            cv2.putText(info_panel, f"Joint MAE:", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 100), 2)
            y_pos += 25
            
            for j in range(min(action_dim, len(joint_errors))):
                err = float(joint_errors[j])
                color = (100, 255, 100) if err < 0.05 else ((255, 255, 100) if err < 0.1 else (255, 100, 100))
                cv2.putText(info_panel, f"  J{j}: {err:.4f}", 
                            (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_pos += 20
        
        combined = np.hstack([frame_bgr, info_panel])  #水平拼接
        writer.write(combined)  #写入一帧
    
    writer.release()
    print(f"  保存视频到: {video_path}")


def save_rollout_results(
    output_dir: Path,
    episode_idx: int,
    predicted_actions: list[np.ndarray],
    ground_truth_actions: list[np.ndarray] = None,
    timestamps: list = None,
    fps: int = 10
):
    """保存rollout结果，包含预测动作、真实动作和误差分析"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_list = [actions.tolist() for actions in predicted_actions]
    gt_list = [actions.tolist() for actions in ground_truth_actions] if ground_truth_actions else None
    
    results = {
        "episode_idx": episode_idx,
        "num_steps": len(predicted_actions),
        "predicted_actions": pred_list,
        "ground_truth_actions": gt_list,
        "timestamps": timestamps or [],
    }
    
    # 计算误差统计
    if ground_truth_actions and len(ground_truth_actions) > 0:
        errors = calculate_error_metrics(predicted_actions, ground_truth_actions)
        results["error_metrics"] = errors
        
        print(f"  Episode {episode_idx} 误差统计:")
        print(f"    平均 MAE: {errors['mean_mae']:.6f}")
        print(f"    平均 MSE: {errors['mean_mse']:.6f}")
        print(f"    最大 MAE: {errors['max_mae']:.6f}")
        print(f"    关节级 MAE: {[f'{x:.4f}' for x in errors['joint_mae']]}")
    
    output_file = output_dir / f"episode_{episode_idx:06d}_actions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  保存动作到: {output_file}")
    return output_file


def calculate_error_metrics(predicted_actions: list, ground_truth_actions: list, action_dim: int = 6) -> dict:
    """计算预测动作与真实动作之间的误差指标"""
    mae_per_step = []
    mse_per_step = []
    
    min_len = min(len(predicted_actions), len(ground_truth_actions))
    
    for i in range(min_len):
        pred = np.asarray(predicted_actions[i]).flatten()[:action_dim]
        gt = np.asarray(ground_truth_actions[i]).flatten()[:action_dim]
        
        mae = np.mean(np.abs(pred - gt))
        mse = np.mean((pred - gt) ** 2)
        
        mae_per_step.append(float(mae))
        mse_per_step.append(float(mse))
    
    pred_array = np.array([np.asarray(a).flatten()[:action_dim] for a in predicted_actions[:min_len]])
    gt_array = np.array([np.asarray(a).flatten()[:action_dim] for a in ground_truth_actions[:min_len]])
    
    joint_mae = np.mean(np.abs(pred_array - gt_array), axis=0)
    
    return {
        "mean_mae": float(np.mean(mae_per_step)),
        "mean_mse": float(np.mean(mse_per_step)),
        "max_mae": float(np.max(mae_per_step)),
        "min_mae": float(np.min(mae_per_step)),
        "std_mae": float(np.std(mae_per_step)),
        "mae_per_step": mae_per_step,
        "mse_per_step": mse_per_step,
        "joint_mae": joint_mae.tolist(),
        "action_dim": action_dim,
    }


def run_rollout(
    policy,
    episode_data,
    video_caps: list[cv2.VideoCapture] = None,
    prompt: str = "pick tennis",
    device: str = "cuda",
    save_actions: bool = True,
    save_images: bool = True,
    output_dir: Path = None,
    episode_idx: int = 0,
) -> dict:
    """执行单个episode的rollout，包含误差评估"""
    
    num_frames = len(episode_data)
    predicted_actions = []
    ground_truth_actions = []
    inference_times = []
    timestamps = []
    
    print(f"\n开始Rollout - Episode {episode_idx}, 共 {num_frames} 帧")
    print("-" * 50)
    
    if hasattr(episode_data, 'to_dict'):
        frames = [episode_data.iloc[i].to_dict() for i in range(len(episode_data))]
    elif isinstance(episode_data, dict):
        num_frames = len(episode_data.get("observation.state", []))
        frames = [{k: v[i] if hasattr(v, '__len__') and len(v) == num_frames else v 
                   for k, v in episode_data.items()} 
                  for i in range(num_frames)]
    else:
        frames = list(episode_data)
    
    try:
        for step, frame in enumerate(frames):
            ######################## 帧的关节数据 #帧数用于读取图片
            obs = format_observation(frame, step, video_caps, prompt)
            
            start_time = time.time()
            with torch.inference_mode():
                action_dict = policy.infer(obs)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            predicted_action = action_dict["actions"]
            if hasattr(predicted_action, 'numpy'):
                predicted_action = predicted_action.numpy()
            predicted_actions.append(predicted_action)
            
            # 收集真实动作
            if "actions" in obs:
                gt_action = obs["actions"]
                if hasattr(gt_action, 'numpy'):
                    gt_action = gt_action.numpy()
                ground_truth_actions.append(gt_action)
            
            if "timestamp" in frame:
                timestamps.append(float(frame["timestamp"]))
            else:
                timestamps.append(step / 10.0)
            
            if step % 10 == 0 or step == len(frames) - 1:
                avg_time = np.mean(inference_times[-10:])
                print(f"  Step {step:4d}/{num_frames}: "
                      f"inference_time={inference_time:.1f}ms "
                      f"(avg={avg_time:.1f}ms)")
        
        total_time = sum(inference_times)
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        print("-" * 50)
        print(f"Rollout完成统计:")
        print(f"  总步数: {num_frames}")
        print(f"  总推理时间: {total_time:.2f}ms")
        print(f"  平均推理时间: {avg_time:.2f} ± {std_time:.2f}ms")
        print(f"  FPS (推理): {1000/avg_time:.2f}")
        
        result = {
            "episode_idx": episode_idx,
            "num_steps": num_frames,
            "predicted_actions": predicted_actions,
            "ground_truth_actions": ground_truth_actions,
            "timestamps": timestamps,
            "inference_times": inference_times,
            "avg_inference_time": avg_time,
            "fps": 1000 / avg_time if avg_time > 0 else 0,
        }
        
        # 计算误差
        if len(ground_truth_actions) > 0:
            errors = calculate_error_metrics(predicted_actions, ground_truth_actions)
            result["error_metrics"] = errors
            print(f"  误差统计 - MAE: {errors['mean_mae']:.6f}, MSE: {errors['mean_mse']:.6f}")
        
        if output_dir is not None:
            if save_actions:
                save_rollout_results(output_dir, episode_idx, predicted_actions, 
                                   ground_truth_actions, timestamps)
            
            if save_images:
                save_visualization(output_dir, episode_idx, frames, predicted_actions,
                                  ground_truth_actions, video_caps[0])
        
        return result
    finally:
        if video_caps[0] is not None:
            release_video(video_caps[0])


def main():
    args = parse_args()
    
    print("=" * 60)
    print("模型 Rollout 评估脚本")
    print("=" * 60)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"设备: {args.device}")
    print(f"推理步数: {args.num_inference_steps}")
    print("=" * 60)
    
    # 1. 加载配置和模型
    print("\n[1/4] 加载配置和模型...")
    cfg = _config.get_config(args.config)
    
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint_dir,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_inference_steps},
    )
    print("模型加载完成")
    
    # 2. 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 运行Rollout
    print(f"\n[2/4] 开始运行Rollout...")
    print(f"评估Episodes: {args.start_episode} ~ {args.start_episode + args.num_episodes - 1}")
    
    all_results = []
    data_dir = Path(args.data_dir)
    
    for ep_idx in range(args.start_episode, args.start_episode + args.num_episodes):
        try:
            # 加载episode数据
            episode_df, video_caps = load_episode_data(data_dir, ep_idx)
            
            # 运行rollout
            result = run_rollout(
                policy=policy,
                episode_data=episode_df,
                video_caps=video_caps,
                prompt="Pick up a package with your left hand, then hand it to your right hand, and use your right hand to place the package on the conveyor belt.",
                device=args.device,
                save_actions=args.save_actions,
                save_images=args.save_images,
                output_dir=output_dir / "visualizations",
                episode_idx=ep_idx,
            )
            all_results.append(result)
            
        except FileNotFoundError as e:
            print(f"警告: Episode {ep_idx} 不存在，跳过 - {e}")
            continue
        except Exception as e:
            print(f"错误: Episode {ep_idx} 处理失败 - {e}")
            continue
    
    # 4. 汇总统计
    print(f"\n[3/4] 汇总统计...")
    print("=" * 60)
    print("所有Episode评估结果:")
    
    total_steps = 0
    total_time = 0
    total_mae = []
    total_mse = []
    
    for result in all_results:
        ep = result["episode_idx"]
        steps = result["num_steps"]
        avg_time = result["avg_inference_time"]
        fps = result["fps"]
        
        print(f"  Episode {ep}: {steps} steps, avg={avg_time:.2f}ms, FPS={fps:.2f}")
        
        if "error_metrics" in result:
            err = result["error_metrics"]
            print(f"    Error - MAE: {err['mean_mae']:.6f}, MSE: {err['mean_mse']:.6f}")
            total_mae.append(err['mean_mae'])
            total_mse.append(err['mean_mse'])
        
        total_steps += steps
        total_time += sum(result["inference_times"])
    
    if all_results:
        overall_fps = total_steps / (total_time / 1000) if total_time > 0 else 0
        print(f"\n总体统计:")
        print(f"  总步数: {total_steps}")
        print(f"  总时间: {total_time/1000:.2f}s")
        print(f"  整体FPS: {overall_fps:.2f}")
        
        if total_mae:
            print(f"  平均MAE: {np.mean(total_mae):.6f} ± {np.std(total_mae):.6f}")
            print(f"  平均MSE: {np.mean(total_mse):.6f}")
    
    # 5. 保存汇总报告
    print(f"\n[4/4] 保存评估报告...")
    
    summary = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "num_episodes": len(all_results),
        "total_steps": total_steps,
        "overall_fps": overall_fps if all_results else 0,
        "overall_error": {
            "mean_mae": float(np.mean(total_mae)) if total_mae else None,
            "std_mae": float(np.std(total_mae)) if total_mae else None,
            "mean_mse": float(np.mean(total_mse)) if total_mse else None,
        } if total_mae else None,
        "results": [
            {
                "episode_idx": r["episode_idx"],
                "num_steps": r["num_steps"],
                "avg_inference_time_ms": r["avg_inference_time"],
                "fps": r["fps"],
                "error_metrics": r.get("error_metrics"),
            }
            for r in all_results
        ],
    }
    
    summary_file = output_dir / "rollout_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"评估报告已保存: {summary_file}")
    print("\nRollout完成!")


if __name__ == "__main__":
    main()
