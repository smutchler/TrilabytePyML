#
#    http://trilabyte.com
#    Trilabyte Python Machine Learning
#    Copyright (c) 2020 - Trilabyte
#    Author: Scott Mutchler
#    Contact: smutchler@trilabyte.com
#

from statistics import mean
from statistics import stdev
import math
import numpy as np
import pandas as pd
from typing import Union

def calcMAPE(x: pd.Series, y: pd.Series) -> float:
    """
    This function takes as input two pandas series and then calculates the
    MAPE (mean absolute percent error) between the two. In order for the MAPE
    to be accurate, the prediction must be passed as the x parameter and the
    actual must be passed as the y parameter. A float is returned that is the 
    mean of the absolute percent errors when comparing x[0]-y[0], x[1]-y[1],
    and so on.
    Parameters
    ----------
    x : pd.Series
        Predicted values
    y : pd.Series
        Actual values
    Returns
    -------
    float
        MAPE of predicted v actuals
    """
    x = x.to_numpy()
    y = y.to_numpy()
    
    pes = []
    
    for idx in range(len(x)):
        if not(math.isnan(x[idx])) and not(math.isnan(y[idx])) and y[idx] != 0.0:
            pe = math.fabs(((x[idx] - y[idx]) / y[idx]) * 100)
            pes.append(pe)

    return mean(pes)

def calcPredictionInterval(x: Union[pd.Series, list]) -> float:
    """
    Computes the positive/negative on a 95% confidence interval for a given
    set of data. First all NaN values are removed, then a standard deviation
    is calculated and returned multiplied by 1.96. The 95% confidence interval
    can then be calculated as the mean of x + the function return as the high
    mark and the mean of x - the function return as the low mark.
    
    NOTE: Technically this can work with any iterable in the place of x due to
    the construction of the filter() function, but I thought it best to limit
    it to pandas series or lists as I'm sure there are lots of other iterables
    that might "technically" work but may not be ideal.
    Parameters
    ----------
    x : Union[pd.Series, list]
        pd.Series or list of values
    Returns
    -------
    float
        Half the width of the confidence interval
    """
    x = list(filter(lambda f: ~np.isnan(f), x)) 
    return 1.96 * stdev(x)