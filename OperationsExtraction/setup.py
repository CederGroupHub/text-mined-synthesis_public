from setuptools import setup, find_packages

setup(name='operations_extractor',
      version='2.9.0',
      description='Operations Extractor',
      url='https://github.com/CederGroupHub/OperationsExtraction',
      author='CederGroup(http://ceder.berkeley.edu)',
      packages=find_packages(),
      install_requires=[
          'gensim',
          'keras',
          'numpy',
          'scipy',
          'spacy>=2.0.0,<2.1.0',
          'tensorflow'
      ],
      zip_safe=False)

setup(name='conditions_extractor',
      version='2.5.0',
      description='Conditions Extractor',
      url='https://github.com/CederGroupHub/OperationsExtraction',
      author='CederGroup(http://ceder.berkeley.edu)',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'spacy>=2.0.0,<2.1.0',
      ],
      zip_safe=False)
