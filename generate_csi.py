"""Generate a time-correlated 2 x 2 Rayleigh trace using a sum of sinusoids."""

import argparse
from pathlib import Path

import numpy as np


def generate(output, n_train, n_test, doppler, seed):
    if min(n_train, n_test) < 1 or doppler <= 0:
        raise ValueError("Use positive split sizes and Doppler frequency")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Both splits belong to one trajectory, sampled once per millisecond.
    time = np.arange(n_train + n_test, dtype=np.float64) * 1e-3
    training = np.lib.format.open_memmap(output / "train.npy", mode="w+", dtype=np.complex64,
                                         shape=(n_train, 2, 2))
    testing = np.lib.format.open_memmap(output / "test.npy", mode="w+", dtype=np.complex64,
                                        shape=(n_test, 2, 2))
    paths = 16
    for rx in range(2):
        for tx in range(2):
            link = np.zeros(len(time), dtype=np.complex128)
            for _ in range(paths):
                amplitude_i, amplitude_q = rng.standard_normal(2)
                angle, phase, offset = rng.uniform(0, np.pi, size=3)
                argument = (2 * np.pi * doppler * time + offset) * np.cos(angle) + phase
                link += amplitude_i * np.cos(argument) + 1j * amplitude_q * np.sin(argument)
            # The original MATLAB generator conjugate-transposes each link into a column.
            link = link.conj() / np.sqrt(paths)
            training[:, rx, tx] = link[:n_train]
            testing[:, rx, tx] = link[n_train:]
    training.flush()
    testing.flush()
    print(f"Wrote {n_train} training and {n_test} test CSI samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "csi")
    parser.add_argument("--train-samples", type=int, default=600000)
    parser.add_argument("--test-samples", type=int, default=120120)
    parser.add_argument("--doppler", type=float, default=100)
    parser.add_argument("--seed", type=int, default=72)
    args = parser.parse_args()
    generate(args.output, args.train_samples, args.test_samples, args.doppler, args.seed)
