# -*- coding: utf-8 -*-
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="rlcode",
    version="0.01",
    description="Reinforcement Learning Code Snippet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KibaAmor/rlcode",
    author="kibaamor",
    author_email="kibaamor@gmail.com",
    keywords="rlcode",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "gym",
        "torch",
        "tensorboard",
        "tqdm",
    ],
    extras_require={"dev": ["pre-commit", "black", "flake8", "pytest"]},
)
