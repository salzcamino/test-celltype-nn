"""Setup script for celltype-nn package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="celltype-nn",
    version="0.1.0",
    description="Deep learning network for cell type prediction from single-cell multi-modal data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/celltype-nn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "scanpy>=1.9.0",
        "anndata>=0.9.0",
        "muon>=0.1.5",
        "mudata>=0.2.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "advanced": [
            "optuna>=3.2.0",
            "onnx>=1.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "celltype-train=celltype_nn.scripts.train:main",
            "celltype-predict=celltype_nn.scripts.predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deep-learning single-cell cell-type scRNA-seq multi-modal",
)
