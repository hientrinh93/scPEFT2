from setuptools import setup, find_packages

setup(
    name="scGPT",
    version="0.1.0",
    description="scGPT: Single Cell Generative Pre-trained Transformer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scanpy",
        "scvi-tools",
        "anndata",
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "wandb",
        "torch",
        "torchtext",
        "loralib",
        "transformers",
    ],
)