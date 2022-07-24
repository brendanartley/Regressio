from setuptools import setup, find_packages

VERSION = '1.0'
PACKAGE_NAME = 'regressio'
AUTHOR = 'Brendan Artley'
AUTHOR_EMAIL = 'brendanartley@gmail.com'
URL = 'https://github.com/brendanartley/regressio'

LICENSE = 'MIT'
DESCRIPTION = 'A python module for regression, interpolation and smoothing.'

INSTALL_REQUIRES = [
      'numpy',
      'matplotlib',
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      python_requires='>=3',
      packages=find_packages()
      )