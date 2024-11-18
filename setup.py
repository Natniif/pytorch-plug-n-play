from pathlib import Path

from setuptools import find_packages, setup

directory = Path(__file__).resolve().parent

setup(
    name="plug",
    packages=find_packages(),
    description="DESCRIPTION_HERE",
    author="Fintan Hardy",
    license="MIT",
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "numpy",
        "scipy",
        "scikit-image",
        "wandb",
        "pyyaml",
        "pillow",
        "fire",
    ],
    extras_require={
        "testing": [
            "pytest",
            "pytest-flake8",
            "pytest-pylint",
            "pytorch-lightning",
            "lightning",
            "torch",
            "pyyaml",
            "torchvision",
            "pillow",
            "numpy",
            "flake8",
            "pylint",
            "mypy",
            "black",
            "isort",
            "pre-commit",
        ]
    },
    python_requires=">=3.6",
)
