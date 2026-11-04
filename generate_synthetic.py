"""Recreate the bundled two-channel sensor dataset using a fixed seed."""

from pathlib import Path
import random


def generate():
    rng = random.Random(1138)
    for split, size in (("calibration", 128), ("development", 192), ("holdout", 256)):
        for index in range(size):
            label = index % 2
            margin = 1 if split == "holdout" and index >= size // 2 else rng.randint(7, 12)
            lower = rng.randint(0, 16 - margin)
            higher = lower + margin
            signals = [higher, lower] if label == 0 else [lower, higher]
            pixels = [*signals, *(rng.randrange(17) for _ in range(62))]
            yield ",".join([split, *map(str, pixels), str(label)])


if __name__ == "__main__":
    path = Path(__file__).with_name("digits-data") / "synthetic.csv"
    path.write_bytes(("\n".join(generate()) + "\n").encode("ascii"))
