from setuptools import setup, find_packages

setup(
    name="quantax",
    version="0.1.0",
    description="Flexible neural quantum states based on QuSpin, JAX, and Equinox",
    author="Ao Chen, Christopher Roth",
    author_email="chenao.phys@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10,<=3.13",
    install_requires=[
        "numpy>=2.0.0",
        "quspin>=1.0.0",
        "matplotlib>=3.8.0",
        "ml_dtypes>=0.4.0",
        "jax>=0.4.34",
        "equinox>=0.11.4",
    ],
)
