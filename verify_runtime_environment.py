#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import os
import platform
from importlib import metadata

import torch


EXPECTED_VERSIONS = {
    "torch": "2.12.1",
    "transformers": "5.14.1",
    "accelerate": "1.14.0",
    "monai": "1.6.0",
    "wandb": "0.28.1",
    "batchgenerators": "0.25.3",
    "batchgeneratorsv2": "0.3.5",
    "nibabel": "5.4.2",
}


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required package is missing: {name}") from exc


def main() -> None:
    print(f"[ENV] Python={platform.python_version()}")
    for package, expected in EXPECTED_VERSIONS.items():
        actual = package_version(package)
        if package == "torch":
            matches = actual.split("+", 1)[0] == expected
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(
                f"{package} version mismatch: expected {expected}, got {actual}"
            )
        print(f"[ENV] {package}={actual}")

    for unused_package in ("torchaudio", "torchvision"):
        if importlib.util.find_spec(unused_package) is not None:
            raise RuntimeError(
                f"{unused_package} must remain uninstalled in this "
                "segmentation runtime."
            )

    if os.environ.get("CHECK_PEFT", "0") == "1":
        peft_version = package_version("peft")
        if peft_version != "0.19.1":
            raise RuntimeError(
                f"peft version mismatch: expected 0.19.1, got {peft_version}"
            )
        print(f"[ENV] peft={peft_version}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")
    if torch.version.cuda != "13.2":
        raise RuntimeError(
            f"Expected PyTorch CUDA runtime 13.2, got {torch.version.cuda}"
        )

    expected_gpus = int(os.environ.get("EXPECTED_GPUS", "4"))
    device_count = torch.cuda.device_count()
    if device_count < expected_gpus:
        raise RuntimeError(
            f"Expected at least {expected_gpus} GPUs, detected {device_count}."
        )
    print(
        f"[ENV] CUDA={torch.version.cuda} cuDNN={torch.backends.cudnn.version()} "
        f"GPU_count={device_count}"
    )

    for index in range(device_count):
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        if major < 12:
            raise RuntimeError(
                f"GPU {index} is not Blackwell-class: {name}, sm_{major}{minor}"
            )
        with torch.cuda.device(index):
            left = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
            right = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
            result = left @ right
            torch.cuda.synchronize()
            if not torch.isfinite(result).all():
                raise RuntimeError(
                    f"GPU {index} BF16 matmul produced non-finite values."
                )
        print(f"[ENV] GPU {index}: {name}, sm_{major}{minor}, BF16=pass")

    print("[PASS] Runtime environment is compatible with the v3a experiments.")


if __name__ == "__main__":
    main()
