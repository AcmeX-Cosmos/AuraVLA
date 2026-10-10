"""Convert committed simulation episodes to a local ACT-ready LeRobot dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .teacher import JOINT_NAMES


def load_episode(path, fps, resolution):
    metadata = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("success") is not True:
        raise ValueError(f"Unsuccessful episode: {path}")
    with np.load(path / "frames.npz", allow_pickle=False) as data:
        state, action, timestamp = data["state"].copy(), data["action"].copy(), data["timestamp"].copy()
    count = len(timestamp)
    if (count < 1 or metadata["frames"] != count or state.shape != (count, 6) or
            action.shape != (count, 6) or timestamp.shape != (count,)):
        raise ValueError(f"Invalid frame counts/shapes: {path}")
    if not all(np.isfinite(values).all() for values in (state, action, timestamp)):
        raise ValueError(f"Nonfinite frame values: {path}")
    if not np.allclose(timestamp, np.arange(count) / fps, atol=1e-6, rtol=0):
        raise ValueError(f"Irregular simulation sampling: {path}")
    images = sorted((path / "rgb").glob("*.png"))
    if [p.name for p in images] != [f"{i:06d}.png" for i in range(count)]:
        raise ValueError(f"Missing/misaligned RGB frames: {path}")
    for image in images:
        with Image.open(image) as img:
            if img.size != tuple(resolution) or img.mode != "RGB":
                raise ValueError(f"Invalid camera image: {image}")
            img.verify()
    return state.astype(np.float32), action.astype(np.float32), images


def export_dataset(raw_root, root, repo_id, *, validate_only=False):
    raw_root, root = Path(raw_root).expanduser().resolve(), Path(root).expanduser().resolve()
    manifest = json.loads((raw_root / "session.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("joint_names") != JOINT_NAMES:
        raise ValueError("Unsupported raw dataset format/joint ordering")
    config = manifest["config"]
    fps, resolution = config["recording"]["fps"], config["camera"]["resolution"]
    episodes = sorted(p for p in raw_root.glob("episode_*") if p.is_dir() and not p.name.endswith(".partial"))
    if not episodes:
        raise ValueError("No committed successful episodes to export")
    if fps <= 0:
        raise ValueError("Dataset FPS must be positive")
    if not validate_only and root.exists():
        raise FileExistsError(f"Export output already exists; use a new directory: {root}")
    # Validate every source episode before creating the LeRobot output directory.
    for episode in episodes:
        load_episode(episode, fps, resolution)
    if validate_only:
        print(f"Validated {len(episodes)} episodes, {fps} FPS, six joints in radians")
        return
    if len(repo_id.split("/")) != 2 or any(not part for part in repo_id.split("/")):
        raise ValueError("repo_id must have the form owner/dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    width, height = resolution
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"{n}.pos" for n in JOINT_NAMES]},
        "action": {"dtype": "float32", "shape": (6,), "names": [f"{n}.pos" for n in JOINT_NAMES]},
        "observation.images.overhead": {"dtype": "video", "shape": (height, width, 3),
                                        "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, root=root, robot_type="so101_sim",
                                   fps=fps, features=features, use_videos=True,
                                   image_writer_threads=2)
    try:
        for episode in episodes:
            state, action, images = load_episode(episode, fps, resolution)
            try:
                for q, command, image in zip(state, action, images):
                    with Image.open(image) as img:
                        rgb = np.array(img, dtype=np.uint8)
                    dataset.add_frame({"observation.state": q, "action": command,
                                       "observation.images.overhead": rgb,
                                       "task": config["recording"]["task"]})
                dataset.save_episode()
            except BaseException:
                dataset.clear_episode_buffer()
                raise
    finally:
        # Required for LeRobot v3 parquet metadata/footer completion.
        dataset.finalize()
    (root / "meta" / "aura_source.json").write_text(json.dumps(
        {"raw_root": str(raw_root), "episodes": [p.name for p in episodes],
         "units": manifest["units"], "config": config}, indent=2), encoding="utf-8")
    print(f"Exported {len(episodes)} successful episodes to {root}; repo_id={repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", default="local/so101_banana")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    export_dataset(args.raw_root, args.root, args.repo_id, validate_only=args.validate_only)


if __name__ == "__main__":
    main()
