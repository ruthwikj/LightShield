from __future__ import annotations

from setuptools import find_packages, setup


setup(
    name="lightshield",
    version="0.1.0",
    description="LightShield: a lightweight prompt injection firewall for LLM applications.",
    author="LightShield",
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests",)),
    install_requires=[],
    extras_require={"dev": ["pytest"]},
)
