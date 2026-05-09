from setuptools import setup, find_packages

setup(
    name="defect-detection",
    version="0.1.0",
    description="Self-Supervised Industrial Defect Detection System",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.3.0",
        "torchmetrics>=1.4.0",
        "timm>=1.0.7",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
    ],
)
