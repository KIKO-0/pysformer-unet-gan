import argparse
from pathlib import Path

import h5py
import numpy as np


def build_h5(
    npy_path: Path,
    out_path: Path,
    train_samples: int,
    test_samples: int,
) -> None:
    data = np.load(npy_path)  # [T, N, H, W]
    if data.ndim != 4:
        raise ValueError(f"Unexpected npy shape: {data.shape}")

    total = data.shape[1]
    need = train_samples + test_samples
    if need > total:
        raise ValueError(f"Requested {need} samples, but only {total} available")

    # Convert to oversampled format expected by precipitation_maps_oversampled_h5:
    # [N, T, H, W]
    samples = np.transpose(data[:, :need], (1, 0, 2, 3)).astype(np.float32) / 255.0
    train_data = samples[:train_samples]
    test_data = samples[train_samples:need]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        train_grp = f.create_group("train")
        test_grp = f.create_group("test")
        train_grp.create_dataset("images", data=train_data, compression="gzip")
        test_grp.create_dataset("images", data=test_data, compression="gzip")

    print(f"Saved: {out_path}")
    print(f"train/images shape: {train_data.shape}")
    print(f"test/images shape: {test_data.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=Path, default=Path("mnist_test_seq.npy"))
    parser.add_argument("--out_path", type=Path, default=Path("data/moving_mnist/moving_mnist_oversampled.h5"))
    parser.add_argument("--train_samples", type=int, default=9000)
    parser.add_argument("--test_samples", type=int, default=1000)
    args = parser.parse_args()

    build_h5(
        npy_path=args.npy_path,
        out_path=args.out_path,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )


if __name__ == "__main__":
    main()

