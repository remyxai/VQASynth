import os
from setuptools import setup, find_packages
from setuptools.command.install import install

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def read_requirements(file):
    with open(file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="vqasynth",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements((this_directory / 'requirements.txt')),
    long_description=long_description,
    long_description_content_type='text/markdown'
)
