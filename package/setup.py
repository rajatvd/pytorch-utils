from setuptools import *

LONG_DESC = """
A bunch of utilities for making training models easier. Also contains useful
modules to make building models easier. Can be used in a jupyter notebook. Also
contains a module which integrates with the package `sacred`.
"""

setup(name='pytorch_utils',
	  version='0.5.4',
	  description='Utilities for training and building models in pytorch',
	  long_description=LONG_DESC,
	  author='Rajat Vadiraj Dwaraknath',
	  url='https://github.com/rajatvd/PytorchUtils',
	  install_requires=['ipython', 'sacred', 'tqdm'],
	  author_email='rajatvd@gmail.com',
	  license='MIT',
	  packages=find_packages(),
	  zip_safe=False)
