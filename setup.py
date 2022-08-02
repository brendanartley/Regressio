from setuptools import setup, find_packages
import pathlib

VERSION = '1.0.1'
PACKAGE_NAME = 'regressio'
AUTHOR = 'Brendan Artley'
AUTHOR_EMAIL = 'brendanartley@gmail.com'
URL = 'https://github.com/brendanartley/regressio'

LICENSE = 'MIT'
DESCRIPTION = 'A python module for regression, interpolation and smoothing.'
LONG_DESCRIPTION = (pathlib.Path(__file__).parent / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'matplotlib',
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      python_requires='>=3.8',
      packages=find_packages()
      )