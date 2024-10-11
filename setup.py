import setuptools
from pathlib import Path

def read_long_description(file="README.md"):
    return Path(file).read_text(encoding='utf-8')

def read_requirements(file="requirements.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setuptools.setup(
    name="vqasynth",
    version="0.0.1",
    author="Remyx AI",
    author_email="contact@remyx.ai",
    description="Compose multimodal datasets",
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url="https://github.com/remyxai/VQASynth",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

