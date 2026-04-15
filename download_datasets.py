"""
download_datasets.py — Pre-download MNIST and FashionMNIST to avoid
internet access issues on HPC compute nodes (e.g. Hercules/CICA).

Run from the login node before submitting SLURM jobs:
    python download_datasets.py [--data-dir ./data/raw]
"""

import argparse
import os
import torchvision
import torchvision.transforms as T

DATASETS = {
    "mnist": torchvision.datasets.MNIST,
    "fashion_mnist": torchvision.datasets.FashionMNIST,
}


def download_all(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    transform = T.ToTensor()

    for name, cls in DATASETS.items():
        for split, train in [("train", True), ("test", False)]:
            print(f"Downloading {name} ({split})...", flush=True)
            cls(data_dir, train=train, download=True, transform=transform)
            print(f"  OK", flush=True)

    print(f"\nDatasets ready in: {os.path.abspath(data_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/raw",
                        help="Directory where datasets will be stored (default: ./data/raw)")
    args = parser.parse_args()
    download_all(args.data_dir)
