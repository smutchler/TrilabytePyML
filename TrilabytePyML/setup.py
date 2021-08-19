from setuptools import setup

setup(name='TrilabytePyML',
      version='1.2',
      description='Trilabyte Python Machine Learning',
      url='http://github.com/smutchler/src',
      author='Scott Mutchler',
      author_email='smutchler@trilabyte.com',
      license='GPLv3',
      packages=['TrilabytePyML','TrilabytePyML.stats','TrilabytePyML.util'],
      install_requires=[
        'pmdarima',
        'loess',
        'scikit-learn',
        'numpy',
        'scipy',
        'pandas',
        'ephem',
        'pystan==2.19.1.1',
        'fbprophet'
      ],
      zip_safe=False)

