import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def parse_line_to_frames(line: str, resize: int) -> np.ndarray | None:
    split_idx = line.find(" ")
    if split_idx == -1:
        return None

    head = line[:split_idx]
    body = line[split_idx:]

    head_parts = head.split(",")
    if len(head_parts) < 3:
        return None

    try:
        first_val = float(head_parts[2])
    except ValueError:
        return None

    body_data = np.fromstring(body, sep=" ", dtype=np.float32)
    radar_data = np.hstack(([first_val], body_data))
    if radar_data.size != 15 * 4 * 101 * 101:
        return None

    radar_data = radar_data.reshape(15, 4, 101, 101)
    radar_img_seq = np.max(radar_data, axis=1)  # [15, 101, 101]

    out = np.empty((15, resize, resize), dtype=np.float32)
    for t in range(15):
        frame = np.clip(radar_img_seq[t], 0, 255).astype(np.uint8)
        frame = Image.fromarray(frame).resize((resize, resize), Image.BILINEAR)
        out[t] = np.asarray(frame, dtype=np.float32) / 255.0
    return out


def convert_txt_to_dataset(
    txt_path: Path,
    group: h5py.Group,
    resize: int,
    chunk_size: int,
    max_samples: int | None,
) -> int:
    ds = group.create_dataset(
        "images",
        shape=(0, 15, resize, resize),
        maxshape=(None, 15, resize, resize),
        dtype=np.float32,
        chunks=(1, 15, resize, resize),
        compression="gzip",
    )

    buffer: list[np.ndarray] = []
    kept = 0
    skipped = 0

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and kept >= max_samples:
                break
            if not line.strip():
                continue

            frames = parse_line_to_frames(line, resize=resize)
            if frames is None:
                skipped += 1
                continue

            buffer.append(frames)
            kept += 1

            if len(buffer) >= chunk_size:
                arr = np.stack(buffer, axis=0)
                old_n = ds.shape[0]
                new_n = old_n + arr.shape[0]
                ds.resize(new_n, axis=0)
                ds[old_n:new_n] = arr
                buffer.clear()

            if kept % 500 == 0:
                print(f"{txt_path.name}: kept={kept}, skipped={skipped}, line={idx + 1}")

    if buffer:
        arr = np.stack(buffer, axis=0)
        old_n = ds.shape[0]
        new_n = old_n + arr.shape[0]
        ds.resize(new_n, axis=0)
        ds[old_n:new_n] = arr

    print(f"{txt_path.name} done: kept={kept}, skipped={skipped}")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", type=Path, required=True)
    parser.add_argument("--test_txt", type=Path, required=True)
    parser.add_argument("--out_h5", type=Path, required=True)
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    args = parser.parse_args()

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, "w") as f:
        train_group = f.create_group("train")
        test_group = f.create_group("test")
        n_train = convert_txt_to_dataset(
            txt_path=args.train_txt,
            group=train_group,
            resize=args.resize,
            chunk_size=args.chunk_size,
            max_samples=args.max_train,
        )
        n_test = convert_txt_to_dataset(
            txt_path=args.test_txt,
            group=test_group,
            resize=args.resize,
            chunk_size=args.chunk_size,
            max_samples=args.max_test,
        )
        print(f"Saved H5: {args.out_h5} | train={n_train}, test={n_test}")


if __name__ == "__main__":
    main()

