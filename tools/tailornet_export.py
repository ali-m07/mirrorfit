"""Research-only bridge from a configured TailorNet checkout to a MirrorFit OBJ.

The TailorNet code, weights, dataset and SMPL files are external inputs. Their
non-commercial licence applies to use of this bridge with those assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def export_mesh(repo: Path, data_dir: Path, weights_dir: Path, inputs: Path,
                output: Path, garment_class: str, gender: str) -> None:
    for path in (repo, data_dir, weights_dir, inputs):
        if not path.exists():
            raise FileNotFoundError(path)
    if not (repo / "models" / "tailornet_model.py").is_file():
        raise ValueError("--repo must point to a TailorNet checkout")

    with np.load(inputs, allow_pickle=False) as sample:
        theta = np.asarray(sample["theta"], dtype=np.float32).reshape(-1)
        beta = np.asarray(sample["beta"], dtype=np.float32).reshape(-1)
        gamma = np.asarray(sample["gamma"], dtype=np.float32).reshape(-1)
    if theta.size != 72 or beta.size == 0 or gamma.size == 0:
        raise ValueError("Expected a 72-value SMPL theta and nonempty beta/gamma")

    sys.path.insert(0, str(repo.resolve()))
    import global_var  # type: ignore[import-not-found]
    global_var.DATA_DIR = str(data_dir.resolve())
    global_var.MODEL_WEIGHTS_PATH = str(weights_dir.resolve())

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The upstream TailorNet runner requires CUDA")
    from models.tailornet_model import get_best_runner  # type: ignore[import-not-found]
    from models.smpl4garment import SMPL4Garment  # type: ignore[import-not-found]
    from utils.rotation import normalize_y_rotation  # type: ignore[import-not-found]

    runner = get_best_runner(garment_class=garment_class, gender=gender)
    smpl = SMPL4Garment(gender=gender)
    normalized = normalize_y_rotation(theta)
    with torch.no_grad():
        displacement = runner.forward(
            thetas=torch.from_numpy(normalized[None]).cuda(),
            betas=torch.from_numpy(beta[None]).cuda(),
            gammas=torch.from_numpy(gamma[None]).cuda(),
        )[0].cpu().numpy()
    _, garment = smpl.run(beta=beta, theta=theta,
                          garment_class=garment_class, garment_d=displacement)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for x, y, z in np.asarray(garment.v):
            fh.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in np.asarray(garment.f):
            fh.write(f"f {a + 1} {b + 1} {c + 1}\n")
    print(json.dumps({"mesh": str(output), "vertices": len(garment.v),
                      "faces": len(garment.f)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="NPZ with theta, beta and gamma arrays")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--garment-class", default="t-shirt")
    parser.add_argument("--gender", choices=("male", "female"), default="male")
    args = parser.parse_args()
    export_mesh(args.repo, args.data_dir, args.weights_dir, args.input,
                args.output, args.garment_class, args.gender)


if __name__ == "__main__":
    main()
