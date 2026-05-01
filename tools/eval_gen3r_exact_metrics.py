from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def load_img(path: Path, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")

    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def psnr_np(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))

    if mse <= 1e-12:
        return float("inf")

    return -10.0 * math.log10(mse)


def ssim_np(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(
        ssim(
            gt,
            pred,
            channel_axis=2,
            data_range=1.0,
        )
    )


class LPIPSEvaluator:
    def __init__(self, device: str):
        import lpips

        self.device = torch.device(device)
        self.model = lpips.LPIPS(net="alex").to(self.device).eval()

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device)

        p = p * 2.0 - 1.0
        g = g * 2.0 - 1.0

        val = self.model(p, g)
        return float(val.item())


def copy_for_fid(
    samples: list[dict],
    fid_root: Path,
    image_size: int | None,
) -> tuple[Path, Path]:
    gt_dir = fid_root / "gt"
    pred_dir = fid_root / "pred"

    if gt_dir.exists():
        shutil.rmtree(gt_dir)

    if pred_dir.exists():
        shutil.rmtree(pred_dir)

    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    k = 0

    for sample in samples:
        gt_paths = [Path(x) for x in sample["gt_paths"]]
        pred_paths = [Path(x) for x in sample.get("pred_paths", [])]

        if len(gt_paths) != len(pred_paths):
            print(
                f"[WARN] skip FID copy for {sample.get('sample_name', '')}: "
                f"gt={len(gt_paths)}, pred={len(pred_paths)}"
            )
            continue

        for gt_path, pred_path in zip(gt_paths, pred_paths):
            gt_img = Image.open(gt_path).convert("RGB")
            pred_img = Image.open(pred_path).convert("RGB")

            if image_size is not None:
                gt_img = gt_img.resize((image_size, image_size), Image.BICUBIC)
                pred_img = pred_img.resize((image_size, image_size), Image.BICUBIC)

            gt_img.save(gt_dir / f"{k:06d}.png")
            pred_img.save(pred_dir / f"{k:06d}.png")
            k += 1

    return gt_dir, pred_dir


def compute_fid(gt_dir: Path, pred_dir: Path, device: str) -> float | None:
    try:
        cmd = [
            "python",
            "-m",
            "pytorch_fid",
            str(gt_dir),
            str(pred_dir),
            "--device",
            device,
        ]

        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )

        text = result.stdout.strip() + "\n" + result.stderr.strip()

        for line in text.splitlines():
            if "FID:" in line:
                return float(line.split("FID:")[-1].strip())

        print(text)
        return None

    except Exception as e:
        print(f"[WARN] FID failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=560)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--compute_fid", action="store_true")

    args = parser.parse_args()

    with args.manifest.open("r") as f:
        samples = json.load(f)

    lpips_eval = LPIPSEvaluator(args.device) if args.compute_lpips else None

    rows = []
    psnrs = []
    ssims = []
    lpipss = []

    for sample in samples:
        gt_paths = [Path(x) for x in sample["gt_paths"]]
        pred_paths = [Path(x) for x in sample.get("pred_paths", [])]

        if len(gt_paths) != len(pred_paths):
            print(
                f"[WARN] skip {sample.get('sample_name', '')}: "
                f"gt={len(gt_paths)}, pred={len(pred_paths)}"
            )
            continue

        for j, (gt_path, pred_path) in enumerate(zip(gt_paths, pred_paths)):
            gt = load_img(gt_path, args.image_size)
            pred = load_img(pred_path, args.image_size)

            psnr_v = psnr_np(pred, gt)
            ssim_v = ssim_np(pred, gt)

            row = {
                "sample_name": sample["sample_name"],
                "scene_id": sample["scene_id"],
                "target_index": int(j),
                "target_view_id": int(sample["target_ids"][j]),
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "psnr": psnr_v,
                "ssim": ssim_v,
            }

            if lpips_eval is not None:
                lpips_v = lpips_eval(pred, gt)
                row["lpips"] = lpips_v
                lpipss.append(lpips_v)

            psnrs.append(psnr_v)
            ssims.append(ssim_v)
            rows.append(row)

    summary = {
        "num_pairs": len(rows),
        "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
        "ssim_mean": float(np.mean(ssims)) if ssims else None,
        "lpips_mean": float(np.mean(lpipss)) if lpipss else None,
    }

    if args.compute_fid:
        fid_root = args.out_json.parent / "fid_images"
        gt_dir, pred_dir = copy_for_fid(samples, fid_root, args.image_size)
        fid = compute_fid(gt_dir, pred_dir, args.device)
        summary["fid"] = fid

    result = {
        "summary": summary,
        "per_image": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    with args.out_json.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to: {args.out_json}")


if __name__ == "__main__":
    main()