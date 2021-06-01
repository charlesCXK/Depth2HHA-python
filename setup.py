from pathlib import Path

from setuptools import find_packages, setup

BASE_PATH = Path(__file__).absolute().parent

requirementPath = BASE_PATH / 'requirements.txt'
install_requires = []

if requirementPath.is_file():
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()
        
setup(
    name='Depth2HHA-python',
    version='0',
    url='https://github.com/skimit/Depth2HHA-python',
    license='MIT License ',
    author='Chen XiaoKang ',
    author_email='pkucxk@pku.edu.cn',
    description='This repo implements HHA-encoding algorithm in python3. HHA image is an encoding '
                'algorithm to make better use of depth images, which was proposed by s-gupta in '
                'this paper: Learning Rich Features from RGB-D Images for Object Detection and '
                'Segmentation.',
    packages=find_packages(),
    install_requires=install_requires,
    )
