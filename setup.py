#!/usr/bin/env python3
"""
Setup script for Quantum Computing 101

This setup script allows the project to be installed as a Python package,
making it easier for users to access utilities and run examples.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from file
def read_requirements(filename):
    with open(os.path.join("examples", filename), "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantum-computing-101",
    version="1.0.0",
    author="Quantum Computing 101 Team",
    author_email="aicomputing101@gmail.com",
    description="A comprehensive quantum computing education platform with 40 hands-on examples",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/AIComputing101/quantum-computing-101",
    project_urls={
        "Bug Tracker": "https://github.com/AIComputing101/quantum-computing-101/issues",
        "Documentation": "https://github.com/AIComputing101/quantum-computing-101/tree/main/docs",
        "Source Code": "https://github.com/AIComputing101/quantum-computing-101",
        "Changelog": "https://github.com/AIComputing101/quantum-computing-101/blob/main/CHANGELOG.md",
    },
    packages=find_packages(where="examples"),
    package_dir={"": "examples"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "full": read_requirements("requirements.txt") + read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "quantum101=utils.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "quantum computing",
        "education",
        "qiskit",
        "quantum algorithms",
        "quantum machine learning",
        "quantum programming",
        "physics",
        "tutorial",
    ],
    zip_safe=False,
)
