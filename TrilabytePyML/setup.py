from setuptools import setup

setup(name='TrilabytePyML',
      version='1.0',
      description='Trilabyte Python Machine Learning',
      url='http://github.com/smutchler/src',
      author='Scott Mutchler',
      author_email='smutchler@trilabyte.com',
      license='GPLv3',
      packages=['TrilabytePyML'],
      install_requires=[
        'pmdarima',
        'loess',
        'scikit-learn',
        'numpy',
        'scipy',
        'pandas',
        'ephem',
        'pystan',
        'fbprophet'
      ],
      zip_safe=False)

