from setuptools import *

long_description = """
A bunch of utilities for making training models easier. Also contains useful modules to make building models easier.
Designed to be used in a jupyter notebook workflow.
"""

setup(name='pytorch_utils',
		version='0.3',
		description='Utilities for training and building models in pytorch',
		long_description=long_description,
		author='Rajat Vadiraj Dwaraknath',
		url='https://github.com/rajatvd/PytorchUtils',
		install_requires=['ipython', 'sacred'],
		author_email='rajatvd@gmail.com',
		license='MIT',
		packages=find_packages(),
		zip_safe=False)