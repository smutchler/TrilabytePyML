# Trilabyte PyML

This is a time series/regression package that handles multiple data sets stacked on top of each other.  

INSTALLATION:

1. This package requires Facebook prophet (fbprophet).  I've only been able to get fbprophet installed using Anaconda so Anaconda is highly recommended.
2. The following required packages should be installed by the pip/conda (under Anaconda) command for TrilabytePyML but ***ONLY IF STEP 3 FAILS*** here are the dependencies:
	a. pip install pandas loess scipy numpy scikit-learn pmdarima
	b. conda install -c anaconda ephem
	c. conda install -c conda-forge pystan fbprophet
3. No proxy:

		pip install git+https://github.com/smutchler/TrilabytePyML.git#"egg=TrilabytePyML&subdirectory=TrilabytePyML"

	If you are behind a proxy try:
	
		pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org git+https://github.com/smutchler/TrilabytePyML.git#"egg=TrilabytePyML&subdirectory=TrilabytePyML"
  

4. look at the examples in the TrilabytePyML/samples folder for usage


NOTES:

The package changed from trilabytePyML to TrilabytePyML.  Please adjust your imports.
