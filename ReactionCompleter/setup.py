from setuptools import setup, find_packages

__author__ = 'Haoyan Huo'
__maintainer__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'

if __name__ == "__main__":
    setup(
        name='ReactionCompleter',
        version="0.0.1",
        python_requires='>=3.4',
        author=__author__,
        author_email=__email__,
        packages=find_packages(),
        zip_safe=False,
        install_requires=[
            'sympy',
        ]
    )
