from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio.v3 as iio
from PIL import Image


def find_video_or_frames(out_dir: Path) -> tuple[Path | None, list[Path]]:
    videos = sorted(
        list(out_dir.rglob("*.mp4"))
        + list(out_dir.rglob("*.avi"))
        + list(out_dir.rglob("*.mov"))
    )

    if videos:
        return videos[0], []

    frames = sorted(
        list(out_dir.rglob("*.png"))
        + list(out_dir.rglob("*.jpg"))
        + list(out_dir.rglob("*.jpeg"))
    )

    # Avoid using input/gt/depth/helper files if they are saved in output_dir.
    frames = [
        p
        for p in frames
        if "input" not in p.name.lower()
        and "gt" not in p.name.lower()
        and "depth" not in p.name.lower()
        and "concat" not in p.name.lower()
    ]

    return None, frames


def extract_predictions(
    out_dir: Path,
    pred_dir: Path,
    num_targets: int,
) -> list[str]:
    pred_dir.mkdir(parents=True, exist_ok=True)

    video_path, frame_paths = find_video_or_frames(out_dir)

    saved = []

    if video_path is not None:
        frames = iio.imread(video_path)

        if frames.ndim != 4:
            raise RuntimeError(
                f"Unexpected video shape from {video_path}: {frames.shape}"
            )

        if frames.shape[0] < num_targets:
            raise RuntimeError(
                f"Video has {frames.shape[0]} frames, "
                f"but num_targets={num_targets}."
            )

        for i in range(num_targets):
            out = pred_dir / f"pred_target_{i:02d}.png"
            Image.fromarray(frames[i]).convert("RGB").save(out)
            saved.append(str(out))

        return saved

    if len(frame_paths) < num_targets:
        raise RuntimeError(
            f"Found only {len(frame_paths)} image frames in {out_dir}, "
            f"but num_targets={num_targets}."
        )

    for i, src in enumerate(frame_paths[:num_targets]):
        out = pred_dir / f"pred_target_{i:02d}.png"
        Image.open(src).convert("RGB").save(out)
        saved.append(str(out))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gen3r_root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--remove_far_points", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    infer_py = args.gen3r_root / "infer.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"Cannot find infer.py: {infer_py}")

    with args.manifest.open("r") as f:
        samples = json.load(f)

    if args.limit > 0:
        samples = samples[: args.limit]

    updated = []

    for i, sample in enumerate(samples):
        out_dir = Path(sample["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(infer_py),
            "--pretrained_model_name_or_path",
            str(args.checkpoint_dir),
            "--task",
            sample["task"],
            "--prompts",
            sample["prompt_path"],
            "--frame_path",
            *sample["frame_paths"],
            "--cameras",
            sample["cameras_path"],
            "--output_dir",
            str(out_dir),
        ]

        if args.remove_far_points:
            cmd.append("--remove_far_points")

        print("=" * 100)
        print(f"[{i + 1}/{len(samples)}] {sample['sample_name']}")
        print(" ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)

            pred_paths = extract_predictions(
                out_dir=out_dir,
                pred_dir=Path(sample["pred_dir"]),
                num_targets=len(sample["target_ids"]),
            )

            sample["pred_paths"] = pred_paths

        updated.append(sample)

    updated_manifest = args.manifest.with_name(
        args.manifest.stem + "_with_preds.json"
    )

    with updated_manifest.open("w") as f:
        json.dump(updated, f, indent=2)

    print(f"Updated manifest: {updated_manifest}")


if __name__ == "__main__":
    main()