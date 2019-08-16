from setuptools import setup, find_packages

setup(name='MaterialParser',
      version='2.0.0',
      description='Synthesis Project',
      url='https://github.com/CederGroupHub/MaterialParser',
      author='CederGroup(http://ceder.berkeley.edu)',
      packages=find_packages(),
      install_requires=[
          'pubchempy',
          'regex',
          'sympy'
      ],
      zip_safe=False)
