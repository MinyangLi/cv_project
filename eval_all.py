#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一评估 ROSE-Benchmark/Benchmark 下 3 个方法（effecterase / propainter / rose）
在 object removal 任务上的视频质量。

说明：
1) GT 在每个类别的 Edited 目录下；
2) 预测在每个类别的 predicted/<method> 目录下；
3) propainter 里的 *_masked_in.mp4 不参与评估（本脚本按 GT 同名匹配，自动忽略）；
4) 指标：PSNR / SSIM / LPIPS(vgg) / VFID；
5) VFID 仅使用 I3D 预训练权重：
   /hpc2hdd/home/mli861/cv_project/ProPainter/weights/i3d_rgb_imagenet.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import lpips
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
PROPAINTER_ROOT = PROJECT_ROOT / "ProPainter"
if str(PROPAINTER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPAINTER_ROOT))

# 仅借用 I3D 网络结构定义，不使用 ProPainter 的 PSNR/SSIM/LPIPS/VFID 计算函数。
from core.metrics import InceptionI3d  # noqa: E402


def read_video_rgb(video_path: Path) -> List[np.ndarray]:
    """
    读取视频为 RGB 帧。

    参数:
        video_path: 视频文件路径。

    返回:
        帧列表，每帧为 uint8 RGB，shape=(H, W, 3)。

    用途:
        作为所有指标（PSNR/SSIM/LPIPS/VFID）的原始输入。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"视频为空: {video_path}")
    return frames


def align_video_frames(
    gt_frames: Sequence[np.ndarray],
    pred_frames: Sequence[np.ndarray],
    max_frames: int | None = 81,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    对齐 GT 与预测视频帧序列。

    参数:
        gt_frames: GT 帧序列。
        pred_frames: 预测帧序列。
        max_frames: 最多参与评估的帧数；None 表示不额外截断。

    返回:
        (gt_aligned, pred_aligned)
        - 长度裁剪到两者最短；
        - 若设置 max_frames，再裁剪到前 max_frames 帧；
        - 分辨率不一致时，将预测帧 resize 到 GT 尺寸。

    用途:
        防止指标计算时出现尺寸或长度不一致。
    """
    n = min(len(gt_frames), len(pred_frames))
    if max_frames is not None:
        n = min(n, int(max_frames))
    gt_out: List[np.ndarray] = []
    pred_out: List[np.ndarray] = []

    for i in range(n):
        gt = gt_frames[i]
        pred = pred_frames[i]
        if pred.shape[:2] != gt.shape[:2]:
            h, w = gt.shape[:2]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
        gt_out.append(gt)
        pred_out.append(pred)

    return gt_out, pred_out


def calc_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """
    计算单帧 PSNR。

    参数:
        img1, img2: 两张 RGB 图像（同 shape）。
        data_range: 像素范围，uint8 图像使用 255。

    返回:
        PSNR 值（越高越好）。

    用途:
        衡量像素级重建误差。
    """
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def calc_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算单帧 MSE。

    参数:
        img1, img2: 两张 RGB 图像（同 shape）。

    返回:
        MSE 值（越低越好）。

    用途:
        衡量像素级平方误差。
    """
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    return float(np.mean((x - y) ** 2))


def calc_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算单帧 MAE。

    参数:
        img1, img2: 两张 RGB 图像（同 shape）。

    返回:
        MAE 值（越低越好）。

    用途:
        衡量像素级绝对误差。
    """
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    return float(np.mean(np.abs(x - y)))


def calc_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """
    计算单帧 SSIM。

    参数:
        img1, img2: 两张 RGB 图像（同 shape）。
        data_range: 像素范围，uint8 图像使用 255。

    返回:
        SSIM 值（越高越好）。

    用途:
        衡量结构相似性与感知质量。
    """
    return float(
        structural_similarity(
            img1,
            img2,
            channel_axis=2,
            data_range=data_range,
        )
    )


def init_lpips_vgg(device: torch.device):
    """
    初始化 LPIPS(vgg) 模型。

    参数:
        device: 运行设备（cuda/cpu）。

    返回:
        LPIPS 模型实例。

    用途:
        计算感知距离 LPIPS（越低越好）。
    """
    model = lpips.LPIPS(net="vgg", verbose=False).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def calc_lpips_frame(
    lpips_model,
    img1: np.ndarray,
    img2: np.ndarray,
    device: torch.device,
) -> float:
    """
    计算单帧 LPIPS(vgg)。

    参数:
        lpips_model: LPIPS 模型。
        img1, img2: 两张 RGB uint8 图像。
        device: 运行设备。

    返回:
        LPIPS 分数（越低越好）。

    用途:
        评估感知层面的相似性。
    """

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        # [0, 255] -> [-1, 1]
        t = (t / 255.0) * 2.0 - 1.0
        return t.to(device)

    with torch.no_grad():
        d = lpips_model(_to_tensor(img1), _to_tensor(img2))
    return float(d.mean().item())


def init_i3d_for_vfid(i3d_weight_path: Path, device: torch.device):
    """
    初始化 VFID 所需 I3D 模型。

    参数:
        i3d_weight_path: I3D 预训练权重路径。
        device: 运行设备。

    返回:
        I3D 模型实例。

    用途:
        从视频中提取时序特征，供 VFID 使用。
    """
    if not i3d_weight_path.exists():
        raise FileNotFoundError(f"I3D 权重不存在: {i3d_weight_path}")

    model = InceptionI3d(400, in_channels=3, final_endpoint="Logits")
    state = torch.load(str(i3d_weight_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def video_frames_to_tensor(frames: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    将视频帧序列转为 I3D 输入张量。

    参数:
        frames: RGB uint8 帧序列。
        device: 运行设备。

    返回:
        张量 shape=(1, T, 3, H, W)，值域 [0,1]。

    用途:
        供 I3D 提取视频级时序特征。
    """
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    ten = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)  # (1,T,3,H,W)
    return ten.to(device)


def extract_i3d_feature(frames: Sequence[np.ndarray], i3d_model, device: torch.device) -> np.ndarray:
    """
    提取单个视频的 I3D 特征向量。

    参数:
        frames: 视频 RGB 帧序列。
        i3d_model: I3D 模型。
        device: 运行设备。

    返回:
        1D numpy 向量（展平特征）。

    用途:
        作为 VFID 的输入特征。
    """
    x = video_frames_to_tensor(frames, device=device)  # (1,T,3,H,W)
    with torch.no_grad():
        feat = i3d_model.extract_features(x.transpose(1, 2), "Logits")
        feat = feat.reshape(feat.shape[0], -1)
    return feat[0].detach().cpu().numpy().astype(np.float64)


def calc_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    计算两个高斯分布的 Frechet Distance。

    参数:
        mu1, sigma1: 分布1均值与协方差。
        mu2, sigma2: 分布2均值与协方差。
        eps: 数值稳定项。

    返回:
        Frechet 距离。

    用途:
        VFID 的核心公式计算。
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def calc_vfid(real_feats: Sequence[np.ndarray], fake_feats: Sequence[np.ndarray], eps: float = 1e-6) -> float:
    """
    计算视频级 VFID。

    参数:
        real_feats: GT 视频的 I3D 特征列表。
        fake_feats: 预测视频的 I3D 特征列表。
        eps: 数值稳定项。

    返回:
        VFID 分数（越低越好）。

    用途:
        衡量真实视频分布与生成视频分布之间的距离。
    """
    if len(real_feats) == 0 or len(fake_feats) == 0:
        raise ValueError("VFID 输入特征为空")

    real = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in real_feats], axis=0)
    fake = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in fake_feats], axis=0)

    mu1, mu2 = real.mean(axis=0), fake.mean(axis=0)

    if real.shape[0] > 1:
        sigma1 = np.cov(real, rowvar=False)
    else:
        sigma1 = np.eye(real.shape[1], dtype=np.float64) * eps

    if fake.shape[0] > 1:
        sigma2 = np.cov(fake, rowvar=False)
    else:
        sigma2 = np.eye(fake.shape[1], dtype=np.float64) * eps

    sigma1 = sigma1 + np.eye(sigma1.shape[0], dtype=np.float64) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0], dtype=np.float64) * eps

    return calc_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)


def evaluate_video_pair(
    gt_video: Path,
    pred_video: Path,
    lpips_model,
    i3d_model,
    device: torch.device,
    max_frames: int | None = 81,
    compute_lpips: bool = True,
    compute_vfid_feature: bool = True,
) -> Dict:
    """
    评估一对视频并返回视频级指标。

    参数:
        gt_video: GT 视频路径。
        pred_video: 预测视频路径。
        lpips_model: LPIPS(vgg) 模型。
        i3d_model: I3D 模型。
        device: 运行设备。
        max_frames: 最多参与评估的帧数。
        compute_lpips: 是否计算 LPIPS。
        compute_vfid_feature: 是否提取 VFID 所需 I3D 特征。

    返回:
        {
          "video": 文件名,
          "num_frames": 帧数,
          "psnr": 视频平均 PSNR,
          "ssim": 视频平均 SSIM,
          "lpips": 视频平均 LPIPS,
          "gt_feat": GT I3D 特征,
          "pred_feat": 预测 I3D 特征
        }

    用途:
        作为“类别×方法”统计的最小评估单元。
    """
    gt_frames = read_video_rgb(gt_video)
    pred_frames = read_video_rgb(pred_video)
    gt_frames, pred_frames = align_video_frames(
        gt_frames,
        pred_frames,
        max_frames=max_frames,
    )

    psnrs: List[float] = []
    ssims: List[float] = []
    mses: List[float] = []
    maes: List[float] = []
    lpips_scores: List[float] = []

    for gt, pred in zip(gt_frames, pred_frames):
        psnrs.append(calc_psnr(gt, pred))
        ssims.append(calc_ssim(gt, pred))
        mses.append(calc_mse(gt, pred))
        maes.append(calc_mae(gt, pred))
        if compute_lpips and lpips_model is not None:
            lpips_scores.append(calc_lpips_frame(lpips_model, gt, pred, device))

    gt_feat = None
    pred_feat = None
    if compute_vfid_feature and i3d_model is not None:
        gt_feat = extract_i3d_feature(gt_frames, i3d_model, device)
        pred_feat = extract_i3d_feature(pred_frames, i3d_model, device)

    return {
        "video": gt_video.name,
        "num_frames": len(gt_frames),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "mse": float(np.mean(mses)),
        "mae": float(np.mean(maes)),
        "lpips": float(np.mean(lpips_scores)) if lpips_scores else None,
        "gt_feat": gt_feat,
        "pred_feat": pred_feat,
    }


def evaluate_method_category(
    benchmark_root: Path,
    category: str,
    method: str,
    lpips_model,
    i3d_model,
    device: torch.device,
    max_frames: int | None = 81,
    compute_lpips: bool = True,
    compute_vfid: bool = True,
) -> Dict:
    """
    评估某类别下某方法的全部视频。

    参数:
        benchmark_root: Benchmark 根目录。
        category: 类别名。
        method: 方法名。
        lpips_model: LPIPS(vgg) 模型。
        i3d_model: I3D 模型。
        device: 运行设备。
        max_frames: 每个视频最多参与评估的帧数。
        compute_lpips: 是否计算 LPIPS。
        compute_vfid: 是否计算 VFID。

    返回:
        类别级结果字典，包含平均 PSNR/SSIM/LPIPS/VFID、缺失文件和逐视频明细。

    用途:
        产出“类别 × 方法”维度结果。
    """
    edited_dir = benchmark_root / category / "Edited"
    pred_dir = benchmark_root / category / "predicted" / method

    gt_videos = sorted(edited_dir.glob("*.mp4"))

    video_results: List[Dict] = []
    missing: List[str] = []
    real_feats: List[np.ndarray] = []
    fake_feats: List[np.ndarray] = []

    for gt_video in gt_videos:
        pred_video = pred_dir / gt_video.name
        # propainter 的 *_masked_in.mp4 没有对应 GT 同名，按同名匹配时自动被忽略
        if not pred_video.exists():
            missing.append(gt_video.name)
            continue

        one = evaluate_video_pair(
            gt_video=gt_video,
            pred_video=pred_video,
            lpips_model=lpips_model,
            i3d_model=i3d_model,
            device=device,
            max_frames=max_frames,
            compute_lpips=compute_lpips,
            compute_vfid_feature=compute_vfid,
        )
        gt_feat = one.pop("gt_feat")
        pred_feat = one.pop("pred_feat")
        if gt_feat is not None and pred_feat is not None:
            real_feats.append(gt_feat)
            fake_feats.append(pred_feat)
        video_results.append(one)

    if len(video_results) == 0:
        return {
            "category": category,
            "method": method,
            "num_videos": 0,
            "missing": missing,
            "avg_psnr": None,
            "avg_ssim": None,
            "avg_mse": None,
            "avg_mae": None,
            "avg_lpips": None,
            "vfid": None,
            "videos": [],
        }

    lpips_values = [x["lpips"] for x in video_results if x["lpips"] is not None]
    vfid_value = float(calc_vfid(real_feats, fake_feats)) if compute_vfid and real_feats else None

    return {
        "category": category,
        "method": method,
        "num_videos": len(video_results),
        "missing": missing,
        "avg_psnr": float(np.mean([x["psnr"] for x in video_results])),
        "avg_ssim": float(np.mean([x["ssim"] for x in video_results])),
        "avg_mse": float(np.mean([x["mse"] for x in video_results])),
        "avg_mae": float(np.mean([x["mae"] for x in video_results])),
        "avg_lpips": float(np.mean(lpips_values)) if lpips_values else None,
        "vfid": vfid_value,
        "videos": video_results,
    }


def summarize_by_method(category_results: Sequence[Dict], method: str) -> Dict:
    """
    汇总某方法在所有类别上的总体结果。

    参数:
        category_results: 全部类别级结果。
        method: 方法名。

    返回:
        方法总体统计字典。

    用途:
        便于横向比较 3 个方法在全 benchmark 上的表现。
    """
    rows = [r for r in category_results if r["method"] == method and r["num_videos"] > 0]
    if not rows:
        return {
            "method": method,
            "num_videos": 0,
            "avg_psnr": None,
            "avg_ssim": None,
            "avg_mse": None,
            "avg_mae": None,
            "avg_lpips": None,
            "avg_vfid": None,
        }

    all_videos = []
    for r in rows:
        all_videos.extend(r["videos"])

    return {
        "method": method,
        "num_videos": len(all_videos),
        "avg_psnr": float(np.mean([x["psnr"] for x in all_videos])),
        "avg_ssim": float(np.mean([x["ssim"] for x in all_videos])),
        "avg_mse": float(np.mean([x["mse"] for x in all_videos if x.get("mse") is not None])),
        "avg_mae": float(np.mean([x["mae"] for x in all_videos if x.get("mae") is not None])),
        "avg_lpips": float(np.mean([x["lpips"] for x in all_videos if x.get("lpips") is not None])) if any(x.get("lpips") is not None for x in all_videos) else None,
        "avg_vfid": float(np.mean([r["vfid"] for r in rows if r["vfid"] is not None])),
    }


def save_csv(category_results: Sequence[Dict], method_summary: Sequence[Dict], out_csv: Path) -> None:
    """
    保存 CSV 汇总。

    参数:
        category_results: 类别级结果。
        method_summary: 方法总体结果。
        out_csv: 输出路径。

    用途:
        便于后续表格查看/论文整理。
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "method", "category", "num_videos", "psnr", "ssim", "mse", "mae", "lpips", "vfid"])
        for r in category_results:
            w.writerow([
                "category",
                r["method"],
                r["category"],
                r["num_videos"],
                r["avg_psnr"],
                r["avg_ssim"],
                r.get("avg_mse"),
                r.get("avg_mae"),
                r["avg_lpips"],
                r["vfid"],
            ])
        for r in method_summary:
            w.writerow([
                "overall",
                r["method"],
                "ALL",
                r["num_videos"],
                r["avg_psnr"],
                r["avg_ssim"],
                r.get("avg_mse"),
                r.get("avg_mae"),
                r["avg_lpips"],
                r["avg_vfid"],
            ])


def update_mse_mae_from_existing_json(
    existing_json: Path,
    benchmark_root: Path,
    methods: Sequence[str],
    categories: Sequence[str],
    device: torch.device,
    max_frames: int | None = 81,
) -> Dict:
    """
    基于已有结果 JSON，仅补算 MSE/MAE，避免重复计算 PSNR/SSIM/LPIPS/VFID。

    参数:
        existing_json: 已有评估结果 JSON 路径。
        benchmark_root: Benchmark 根目录。
        methods: 方法列表。
        categories: 类别列表。
        device: 运行设备。
        max_frames: 每个视频最多参与评估的帧数。

    返回:
        合并后的完整结果字典（保留旧指标并新增 mse/mae）。

    用途:
        在已有评估基础上快速增量补指标。
    """
    if not existing_json.exists():
        raise FileNotFoundError(f"existing_json 不存在: {existing_json}")

    with existing_json.open("r", encoding="utf-8") as f:
        existing = json.load(f)

    existing_category_map = {
        (r.get("category"), r.get("method")): r
        for r in existing.get("category_results", [])
    }

    existing_video_map: Dict[Tuple[str, str, str], Dict] = {}
    for r in existing.get("category_results", []):
        c = r.get("category")
        m = r.get("method")
        for v in r.get("videos", []):
            existing_video_map[(c, m, v.get("video"))] = v

    category_results: List[Dict] = []

    for category in categories:
        for method in methods:
            print(f"\n[Update MSE/MAE] category={category}, method={method}")
            fresh = evaluate_method_category(
                benchmark_root=benchmark_root,
                category=category,
                method=method,
                lpips_model=None,
                i3d_model=None,
                device=device,
                max_frames=max_frames,
                compute_lpips=False,
                compute_vfid=False,
            )

            old = existing_category_map.get((category, method), {})
            merged_videos = []
            for v in fresh.get("videos", []):
                old_v = existing_video_map.get((category, method, v.get("video")), {})
                merged = dict(old_v)
                merged.update(v)
                merged_videos.append(merged)

            merged_row = dict(old)
            merged_row.update({
                "category": category,
                "method": method,
                "num_videos": fresh.get("num_videos", 0),
                "missing": fresh.get("missing", []),
                "avg_mse": fresh.get("avg_mse"),
                "avg_mae": fresh.get("avg_mae"),
                "videos": merged_videos,
            })

            # 若旧结果缺失这些字段，也补齐
            merged_row.setdefault("avg_psnr", old.get("avg_psnr"))
            merged_row.setdefault("avg_ssim", old.get("avg_ssim"))
            merged_row.setdefault("avg_lpips", old.get("avg_lpips"))
            merged_row.setdefault("vfid", old.get("vfid"))

            category_results.append(merged_row)

            print(
                f"  videos={merged_row['num_videos']}, "
                f"MSE={merged_row.get('avg_mse')}, MAE={merged_row.get('avg_mae')}"
            )

    method_summary = [summarize_by_method(category_results, m) for m in methods]

    result = dict(existing)
    result["benchmark_root"] = str(benchmark_root)
    result["categories"] = list(categories)
    result["methods"] = list(methods)
    result["category_results"] = category_results
    result["method_summary"] = method_summary
    result["updated"] = {"mse": True, "mae": True}

    return result


def main() -> None:
    """
    主函数。

    参数（命令行）:
        --benchmark_root: Benchmark 根目录。
        --methods: 参与评估的方法列表。
        --categories: 可选，指定类别；默认自动扫描。
        --i3d_model_path: VFID 使用的 I3D 权重路径。
        --device: cuda/cpu。
        --out_json: 输出详细 JSON。
        --out_csv: 输出汇总 CSV。

    用途:
        一次性完成 3 个方法在全 benchmark 的评估与结果保存。
    """
    parser = argparse.ArgumentParser("Evaluate object-removal videos for ROSE-Benchmark")
    parser.add_argument(
        "--benchmark_root",
        type=str,
        default=str(PROJECT_ROOT / "ROSE-Benchmark" / "Benchmark"),
        help="Benchmark 根目录",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["effecterase", "propainter", "rose"],
        help="predicted 下的方法目录名",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="可选：手动指定类别；默认自动扫描",
    )
    parser.add_argument(
        "--i3d_model_path",
        type=str,
        default=str(PROJECT_ROOT / "ProPainter" / "weights" / "i3d_rgb_imagenet.pt"),
        help="VFID 用 I3D 预训练权重路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="运行设备",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "eval_all_results.json"),
        help="详细结果 JSON 路径",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "eval_all_results.csv"),
        help="汇总结果 CSV 路径",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=81,
        help="每个视频最多参与评估的帧数（默认 81）",
    )
    parser.add_argument(
        "--update_mse_mae_only",
        action="store_true",
        help="仅补算 MSE/MAE；其余指标从已有 JSON 读取",
    )
    parser.add_argument(
        "--existing_json",
        type=str,
        default=None,
        help="增量模式下已有结果 JSON 路径；默认使用 --out_json",
    )
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).resolve()
    i3d_model_path = Path(args.i3d_model_path).resolve()
    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    device = torch.device(args.device)

    if not benchmark_root.exists():
        raise FileNotFoundError(f"benchmark_root 不存在: {benchmark_root}")

    if args.categories is None:
        categories = sorted([p.name for p in benchmark_root.iterdir() if p.is_dir()])
    else:
        categories = args.categories

    print(f"[Info] benchmark_root: {benchmark_root}")
    print(f"[Info] categories: {categories}")
    print(f"[Info] methods: {args.methods}")
    print(f"[Info] device: {device}")

    if args.update_mse_mae_only:
        existing_json = Path(args.existing_json).resolve() if args.existing_json else out_json
        print(f"[Info] update_mse_mae_only=True, existing_json={existing_json}")
        result = update_mse_mae_from_existing_json(
            existing_json=existing_json,
            benchmark_root=benchmark_root,
            methods=args.methods,
            categories=categories,
            device=device,
            max_frames=args.max_frames,
        )
        method_summary = result["method_summary"]
        category_results = result["category_results"]
    else:
        lpips_model = init_lpips_vgg(device)
        i3d_model = init_i3d_for_vfid(i3d_model_path, device)

        category_results: List[Dict] = []
        for category in categories:
            for method in args.methods:
                print(f"\n[Eval] category={category}, method={method}")
                one = evaluate_method_category(
                    benchmark_root=benchmark_root,
                    category=category,
                    method=method,
                    lpips_model=lpips_model,
                    i3d_model=i3d_model,
                    device=device,
                    max_frames=args.max_frames,
                    compute_lpips=True,
                    compute_vfid=True,
                )
                category_results.append(one)
                print(
                    f"  videos={one['num_videos']}, "
                    f"PSNR={one['avg_psnr']}, SSIM={one['avg_ssim']}, "
                    f"MSE={one.get('avg_mse')}, MAE={one.get('avg_mae')}, "
                    f"LPIPS={one['avg_lpips']}, VFID={one['vfid']}, missing={len(one['missing'])}"
                )

        method_summary = [summarize_by_method(category_results, m) for m in args.methods]

        result = {
            "benchmark_root": str(benchmark_root),
            "categories": categories,
            "methods": args.methods,
            "lpips_net": "vgg",
            "i3d_model_path": str(i3d_model_path),
            "category_results": category_results,
            "method_summary": method_summary,
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    save_csv(category_results, method_summary, out_csv)

    print("\n================ Final Summary ================")
    for row in method_summary:
        print(
            f"{row['method']:12s} | videos={row['num_videos']:3d} | "
            f"PSNR={row['avg_psnr']} | SSIM={row['avg_ssim']} | "
            f"MSE={row.get('avg_mse')} | MAE={row.get('avg_mae')} | "
            f"LPIPS={row['avg_lpips']} | VFID(mean)={row['avg_vfid']}"
        )
    print(f"[Saved] JSON: {out_json}")
    print(f"[Saved] CSV : {out_csv}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
