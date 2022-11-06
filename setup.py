import os
import subprocess
from pathlib import Path
from setuptools import find_packages, setup

import torch

PROJECT_NAME = "dl-playground"
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
DESCRIPTION = "Deep Learning Playground"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 13], "Requires PyTorch >= 1.13"


def get_version() -> str:
    cwd = Path(__file__).parent

    with open(cwd / "version.txt") as f:
        version = f.readline().strip()

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except Exception:
        sha = "Unknown"

    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    # write version.py
    with open(cwd / PACKAGE_NAME / "version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")

    return version


if __name__ == "__main__":
    version = get_version()

    print(f"Building {PROJECT_NAME}-{version}")

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Ming Yang",
        author_email="ymviv@qq.com",
        url=f"https://github.com/vivym/{PROJECT_NAME}",
        download_url=f"https://github.com/vivym/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.9",
        install_requires=[
            "pytorch_lightning",
            "jsonargparse[signatures]",
            "wandb",
            "rich",
            "einops",
            "torchmetrics",
            "torch-scatter",
            "torch-sparse",
            "torch-geometric",
         ],
    )
