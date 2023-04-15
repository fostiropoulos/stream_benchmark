#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="stream_benchmark",
    version="1.0",
    description="Stream Benchmark",
    author="Iordanis Fostiropoulos",
    author_email="dev@iordanis.xyz",
    url="https://iordanis.xyz/",
    python_requires=">3.10",
    long_description=open("README.md").read(),
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "torch==2.0.0",
        "torchvision==0.15.1",
        "numpy==1.24.2",
        "ray==2.3.1",
        "setproctitle==1.3.2",
        "quadprog==0.1.11",
        "pandas==2.0.0",
        "tabulate==0.9.0",
        "https://github.com/fostiropoulos/stream.git",
    ],
    extras_require={
        "dev": [
            "mypy==1.2.0",
            "pytest==7.3.0",
            "pylint==2.17.2",
            "flake8==6.0.0",
            "black==23.3.0",
            "types-requests==2.28.11.17",
        ],
    },
)
