"""
setup.py
========
Package installation configuration for the
Hybrid Credit Scoring project.

Usage
-----
# Development install (editable mode — allows src/ imports)
    pip install -e .

# Standard install
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for PyPI long description
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="hybrid-credit-scoring-vietnam",
    version="1.0.0",
    description=(
        "Hybrid Credit Scoring for Gen Z Thin-File Customers in Vietnam: "
        "Integrating CIC Bureau Data with Digital Behavioral Signals via XGBoost"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Phạm Tiến Dũng, Phạm Trường Phát, Phạm Thanh Huyền",
    author_email="",
    url="https://github.com/[YOUR_USERNAME]/hybrid-credit-scoring-vietnam",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "imbalanced-learn>=0.11.0",
        "optuna>=3.4.0",
        "shap>=0.43.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
        "loguru>=0.7.0",
        "scorecardpy>=0.1.9.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    keywords=[
        "credit-risk",
        "credit-scoring",
        "xgboost",
        "behavioral-data",
        "fintech",
        "vietnam",
        "machine-learning",
        "quantitative-finance",
    ],
)
