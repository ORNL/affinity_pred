#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='affinity_pred',
      packages=['affinity_pred'],
      version=1.0,
      description='Protein-ligand affinity prediction',
      author='Jens Glaser',
      author_email='glaserj@ornl.gov',
      url='https://github.com/jglaser/affinity_pred',
      license='BSD',
      keywords=['affinity prediction','transformers','NLP','SMILES','pytorch'],
      install_requires=[
          'transformers',
          'datasets',
          'torch',
      ],
      )
