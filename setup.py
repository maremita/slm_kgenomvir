from setuptools import setup, find_packages
from lm_genomvir import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='lm_genomvir',
    version=_version,
    description='evaluation of statistical Linear Models for GENOMe VIRus classification',
    author='remita',
    author_email='amine.m.remita@gmail.com',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES
)
