from pkg_resources import parse_requirements
from setuptools import setup, find_packages

setup(name='pgg',
      version='1.0.1',
      packages=find_packages(),
      install_reqs=parse_requirements('requirements.txt')
)
