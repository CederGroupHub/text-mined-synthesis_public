from setuptools import setup, find_packages

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

if __name__ == "__main__":
    setup(name='materials_entity_recognition',
          version=2.0,
          author="Tanjin He",
          author_email="tanjin_he@berkeley.edu",
          license="MIT License",
          packages=find_packages(),
          install_requires=[
              'tensorflow-gpu>=2.1',
              'tensorflow-addons>=0.8',
          ],
          zip_safe=False)
