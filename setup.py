import setuptools
from setuptools.command.install import install
import subprocess
import sys
from pathlib import Path

def read_long_description(file="README.md"):
    return Path(file).read_text(encoding='utf-8')

def read_requirements(file="requirements.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and 'git+' not in line]

class CustomInstallCommand(install):
    """Custom installation script to install VCS dependencies before the package."""

    def run(self):
        # List of VCS dependencies
        vcs_dependencies = [
            "git+https://github.com/zhijian-liu/torchpack.git@3a5a9f7ac665444e1eb45942ee3f8fc7ffbd84e5",
            "git+https://github.com/alibaba/TinyNeuralNetwork.git",
            "git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588"
        ]

        # Install each VCS dependency using pip
        for dep in vcs_dependencies:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

        # Proceed with the standard setuptools installation
        install.run(self)

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
    python_requires='>=3.8',
    cmdclass={
        'install': CustomInstallCommand,
    }
)

