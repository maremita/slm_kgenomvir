from setuptools import setup, find_packages
from slm_kgenomvir import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='slm_kgenomvir',
    version=_version,
    description='Evaluation of statistical linear models'+\
            ' for kmer-based genome virus classification',
    author='remita',
    author_email='remita.amine@courrier.uqam.ca',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES
)
