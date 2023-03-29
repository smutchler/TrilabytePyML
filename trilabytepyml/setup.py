#from setuptools import setup
from distutils.core import setup

setup(name='trilabytepyml',
      version='3.2',
      description='Trilabyte Python Machine Learning',
      url='http://github.com/smutchler/src',
      author='Scott Mutchler',
      author_email='smutchler@trilabyte.com',
      license='GPLv3',
      packages=['trilabytepyml','trilabytepyml.stats','trilabytepyml.util'],
      install_requires=[
        'pmdarima',
        'loess',
        'scikit-learn>=1.1.2',
        'numpy',
        'scipy',
        'pandas'
      ],
      zip_safe=False)

