# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from setuptools import setup, find_packages

setup(
    name='uda_helpers',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'torch>=1.7',
        'torchvision>=0.8',
        'matplotlib',
        'pyyaml',
        'prettytable',
    ],
)