from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def select_context_indices(
    source_sequence: list[int],
    n: int,
    mode: str,
) -> list[int]:
    source = list(source_sequence)

    if len(source) == 0:
        return []

    if n <= 0 or n >= len(source):
        return source

    if mode == "first":
        return source[:n]

    if mode == "last":
        return source[-n:]

    if mode == "center":
        start = max((len(source) - n) // 2, 0)
        return source[start : start + n]

    if mode == "even":
        positions = np.linspace(0, len(source) - 1, n)
        return [source[int(round(p))] for p in positions]

    raise ValueError(f"Unknown context_selection={mode}")


def find_image(color_dir: Path, view_id: int) -> Path:
    candidates = [
        color_dir / f"frame_{view_id:04d}.jpg",
        color_dir / f"frame_{view_id:04d}.png",
        color_dir / f"frame_{view_id:04d}.jpeg",
        color_dir / f"frame_{view_id:05d}.jpg",
        color_dir / f"frame_{view_id:05d}.png",
        color_dir / f"frame_{view_id:05d}.jpeg",
        color_dir / f"frame_{view_id:06d}.jpg",
        color_dir / f"frame_{view_id:06d}.png",
        color_dir / f"frame_{view_id:06d}.jpeg",
        color_dir / f"{view_id}.jpg",
        color_dir / f"{view_id}.png",
        color_dir / f"{view_id}.jpeg",
        color_dir / f"{view_id:04d}.jpg",
        color_dir / f"{view_id:04d}.png",
        color_dir / f"{view_id:04d}.jpeg",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find image for view_id={view_id} in {color_dir}. "
        f"Tried frame_{view_id:04d}.jpg and related patterns."
    )


def find_pose(pose_dir: Path, view_id: int) -> Path:
    candidates = [
        pose_dir / f"frame_{view_id:04d}.txt",
        pose_dir / f"frame_{view_id:04d}.npy",
        pose_dir / f"frame_{view_id:05d}.txt",
        pose_dir / f"frame_{view_id:05d}.npy",
        pose_dir / f"frame_{view_id:06d}.txt",
        pose_dir / f"frame_{view_id:06d}.npy",
        pose_dir / f"{view_id}.txt",
        pose_dir / f"{view_id}.npy",
        pose_dir / f"{view_id:04d}.txt",
        pose_dir / f"{view_id:04d}.npy",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find pose for view_id={view_id} in {pose_dir}. "
        f"Tried frame_{view_id:04d}.txt and related patterns."
    )


def find_intrinsic(intrinsic_dir: Path) -> Path:
    candidates = [
        intrinsic_dir / "intrinsic_depth.txt",
        intrinsic_dir / "intrinsic_color.txt",
        intrinsic_dir / "intrinsic.txt",
        intrinsic_dir / "intrinsics.txt",
        intrinsic_dir / "K.txt",
        intrinsic_dir / "camera.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find intrinsic file in {intrinsic_dir}. "
        f"Expected intrinsic_depth.txt or similar."
    )


def load_pose(path: Path, invert_pose: bool = False) -> np.ndarray:
    if path.suffix == ".npy":
        pose = np.load(path).astype(np.float32)
    else:
        pose = np.loadtxt(path).astype(np.float32)

    if pose.size == 16:
        pose = pose.reshape(4, 4)

    if pose.shape != (4, 4):
        raise ValueError(f"Unsupported pose shape: {path}, shape={pose.shape}")

    if invert_pose:
        pose = np.linalg.inv(pose)

    return pose.astype(np.float32)


def load_intrinsic(path: Path, image_w: int, image_h: int) -> np.ndarray:
    data = np.loadtxt(path).astype(np.float32)

    if data.ndim == 1:
        if data.size == 4:
            fx, fy, cx, cy = data.tolist()
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        elif data.size == 6:
            fx, fy, cx, cy, _, _ = data.tolist()
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        elif data.size == 9:
            K = data.reshape(3, 3).astype(np.float32)

        elif data.size == 16:
            K = data.reshape(4, 4)[:3, :3].astype(np.float32)

        else:
            raise ValueError(
                f"Unsupported intrinsic format: {path}, shape={data.shape}"
            )

    elif data.shape == (3, 3):
        K = data.astype(np.float32)

    elif data.shape == (4, 4):
        K = data[:3, :3].astype(np.float32)

    else:
        raise ValueError(f"Unsupported intrinsic format: {path}, shape={data.shape}")

    # If K is normalized, convert to pixel-space first.
    if K[0, 0] < 10.0 and K[1, 1] < 10.0:
        K = K.copy()
        K[0, 0] *= image_w
        K[0, 2] *= image_w
        K[1, 1] *= image_h
        K[1, 2] *= image_h

    return K.astype(np.float32)


def update_K_for_resize_center_crop(
    K: np.ndarray,
    image_w: int,
    image_h: int,
    out_size: int = 560,
) -> np.ndarray:
    """
    Match Gen3R image preprocessing:
      resize so the shorter side becomes out_size,
      then center-crop to out_size x out_size.
    """
    scale = out_size / float(min(image_w, image_h))

    new_w = image_w * scale
    new_h = image_h * scale

    K2 = K.astype(np.float32).copy()
    K2[0, 0] *= scale
    K2[1, 1] *= scale
    K2[0, 2] *= scale
    K2[1, 2] *= scale

    crop_x = max((new_w - out_size) / 2.0, 0.0)
    crop_y = max((new_h - out_size) / 2.0, 0.0)

    K2[0, 2] -= crop_x
    K2[1, 2] -= crop_y

    return K2.astype(np.float32)


def symlink_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt.strip() + "\n")


def make_camera_json(
    scene_dir: Path,
    color_dir_name: str,
    pose_dir_name: str,
    intrinsic_dir_name: str,
    context_ids: list[int],
    target_ids: list[int],
    out_path: Path,
    invert_pose: bool,
    gen3r_size: int,
) -> None:
    """
    Write Gen3R camera JSON.

    Your local Gen3R/infer.py expects:

        cameras["extrinsics"]
        cameras["intrinsics"]

    Here we put ONLY exact target cameras into these arrays.

    Example:
        target_ids = [43, 44]

    Then:
        len(extrinsics) = 2
        len(intrinsics) = 2

    This lets Gen3R generate exact target views instead of using "free".
    """

    color_dir = scene_dir / color_dir_name
    pose_dir = scene_dir / pose_dir_name
    intrinsic_dir = scene_dir / intrinsic_dir_name

    # Use one image to infer original image size.
    first_img = Image.open(find_image(color_dir, context_ids[0])).convert("RGB")
    image_w, image_h = first_img.size

    K_raw = load_intrinsic(find_intrinsic(intrinsic_dir), image_w, image_h)
    K_560 = update_K_for_resize_center_crop(
        K_raw,
        image_w=image_w,
        image_h=image_h,
        out_size=gen3r_size,
    )

    extrinsics = []
    intrinsics = []

    for view_id in target_ids:
        pose = load_pose(find_pose(pose_dir, view_id), invert_pose=invert_pose)

        extrinsics.append(pose.astype(np.float32).tolist())
        intrinsics.append(K_560.astype(np.float32).tolist())

    data = {
        # Required by Gen3R infer.py.
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,

        # Extra metadata for our own evaluation scripts.
        "target_view_ids": [int(x) for x in target_ids],
        "context_view_ids": [int(x) for x in context_ids],
        "height": int(gen3r_size),
        "width": int(gen3r_size),
        "h": int(gen3r_size),
        "w": int(gen3r_size),
        "camera_model": "OPENCV",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)


def build_sample(
    root: Path,
    scene_id: str,
    window_index: int,
    difficulty: str,
    item: dict[str, Any],
    out_root: Path,
    num_context_views: int,
    context_selection: str,
    prompt: str,
    copy_images: bool,
    color_dir_name: str,
    pose_dir_name: str,
    intrinsic_dir_name: str,
    invert_pose: bool,
    gen3r_size: int,
) -> dict[str, Any]:
    scene_dir = root / scene_id
    color_dir = scene_dir / color_dir_name

    context_ids = select_context_indices(
        item["source_sequence"],
        n=num_context_views,
        mode=context_selection,
    )
    target_ids = list(item["targets"])

    if len(context_ids) == 1:
        task = "1view"
    elif len(context_ids) == 2:
        task = "2view"
    else:
        raise ValueError(
            "Gen3R official infer.py supports 1view/2view/allview. "
            "This exact-target wrapper supports 1view or 2view. "
            f"Got context_ids={context_ids}."
        )

    sample_name = (
        f"{scene_id}_w{window_index}_{difficulty}"
        f"_ctx{'-'.join(map(str, context_ids))}"
        f"_tgt{'-'.join(map(str, target_ids))}"
    )

    sample_dir = out_root / sample_name
    frames_dir = sample_dir / "frames"

    frame_paths = []
    for i, view_id in enumerate(context_ids):
        src = find_image(color_dir, view_id)
        dst = frames_dir / f"context_{i:02d}_frame_{view_id:04d}{src.suffix.lower()}"
        symlink_or_copy(src, dst, copy=copy_images)
        frame_paths.append(str(dst))

    gt_dir = sample_dir / "gt_targets"
    gt_paths = []
    for i, view_id in enumerate(target_ids):
        src = find_image(color_dir, view_id)
        dst = gt_dir / f"target_{i:02d}_frame_{view_id:04d}.png"
        gt_dir.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").save(dst)
        gt_paths.append(str(dst))

    prompt_path = sample_dir / "prompts.txt"
    write_prompt(prompt_path, prompt)

    cameras_path = sample_dir / "cameras_exact_targets.json"
    make_camera_json(
        scene_dir=scene_dir,
        color_dir_name=color_dir_name,
        pose_dir_name=pose_dir_name,
        intrinsic_dir_name=intrinsic_dir_name,
        context_ids=context_ids,
        target_ids=target_ids,
        out_path=cameras_path,
        invert_pose=invert_pose,
        gen3r_size=gen3r_size,
    )

    meta = {
        "sample_name": sample_name,
        "scene_id": scene_id,
        "window_index": int(window_index),
        "difficulty": difficulty,
        "task": task,
        "context_ids": context_ids,
        "target_ids": target_ids,
        "prompt_path": str(prompt_path),
        "frame_paths": frame_paths,
        "gt_paths": gt_paths,
        "cameras_path": str(cameras_path),
        "output_dir": str(sample_dir / "gen3r_output"),
        "pred_dir": str(sample_dir / "pred_targets"),
        "gt_dir": str(gt_dir),
    }

    with (sample_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval_json", type=str, default="re10k_eval_v2.json")
    parser.add_argument("--out_root", type=Path, required=True)

    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--num_context_views", type=int, default=2)
    parser.add_argument("--context_selection", type=str, default="even")
    parser.add_argument("--max_samples", type=int, default=-1)

    parser.add_argument("--prompt", type=str, default="a realistic indoor room")

    parser.add_argument("--color_dir_name", type=str, default="color")
    parser.add_argument("--pose_dir_name", type=str, default="pose")
    parser.add_argument("--intrinsic_dir_name", type=str, default="intrinsic")

    parser.add_argument("--invert_pose", action="store_true")
    parser.add_argument("--gen3r_size", type=int, default=560)
    parser.add_argument("--copy_images", action="store_true")

    args = parser.parse_args()

    eval_path = args.root / args.eval_json
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval json not found: {eval_path}")

    with eval_path.open("r") as f:
        eval_data = json.load(f)

    args.out_root.mkdir(parents=True, exist_ok=True)

    all_meta = []
    count = 0

    for scene_id, windows in eval_data.items():
        scene_dir = args.root / scene_id
        if not scene_dir.exists():
            print(f"[WARN] Missing scene dir: {scene_dir}")
            continue

        for window in windows:
            window_index = int(window.get("window_index", -1))
            item = window.get(args.difficulty, None)

            if item is None:
                continue

            if "source_sequence" not in item or "targets" not in item:
                continue

            meta = build_sample(
                root=args.root,
                scene_id=scene_id,
                window_index=window_index,
                difficulty=args.difficulty,
                item=item,
                out_root=args.out_root,
                num_context_views=args.num_context_views,
                context_selection=args.context_selection,
                prompt=args.prompt,
                copy_images=args.copy_images,
                color_dir_name=args.color_dir_name,
                pose_dir_name=args.pose_dir_name,
                intrinsic_dir_name=args.intrinsic_dir_name,
                invert_pose=args.invert_pose,
                gen3r_size=args.gen3r_size,
            )

            all_meta.append(meta)
            count += 1

            if args.max_samples > 0 and count >= args.max_samples:
                break

        if args.max_samples > 0 and count >= args.max_samples:
            break

    manifest = args.out_root / "manifest_exact.json"
    with manifest.open("w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"Prepared {len(all_meta)} samples.")
    print(f"Manifest: {manifest}")

    if len(all_meta) > 0:
        print("First sample:")
        print(json.dumps(all_meta[0], indent=2))


if __name__ == "__main__":
    main()