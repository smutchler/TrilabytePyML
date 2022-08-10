from typing import Union

def getParam(param: str, options: dict) -> Union[str, int, float, list, bool]:
    """
    This function returns the value stored in the dictionary "options" under
    the key "param". If the user has set a value, it returns the user value,
    otherwise it returns the default value, specified in this function. The
    output can be a string, integer, float, list, or boolean, depending on the
    key specified. The function is set up such that if an incorrectly
    constructed options dictionary is passed, it reverts to defaults.
    Parameters
    ----------
    param : str
        Key of the options dictionary for which you want the value
    options : dict
        Dictionary of parameters to eventually be passed to the forecast
        algorithms
    Returns
    -------
    Union[str, int, float, list, dict, bool]
        Value corresponding to the "param" key in the "optinos" dictionary
    """
    val = None if not(param in options) else options[param]
    
    if (val == None):
        defaults = dict()
        defaults['sortColumns'] = []
        defaults['splitColumns'] = []
        defaults['predictorColumns'] = []
        defaults['targetColumn'] = None 
        defaults['periodicity'] = 12
        defaults['seasonality'] = 'Auto' #'Auto'  # 'Auto','None','Additive','Multiplicative' 
        defaults['method'] = 'Auto' #'Auto','ARIMA','MLR'
        defaults['autoDetectOutliers'] = True
        defaults['outlierStdevMultiplier'] = 3.0
        defaults['seasonalityBandwidth'] = 0.7
        defaults['ridgeAlpha'] = 1.0
        defaults['forceNonNegative'] = False
        defaults['scalePredictors'] = False 
        # fraction is for Ridge / rows is for Foreecast
        defaults['holdoutFraction'] = 0.0
        defaults['numHoldoutRows'] = 0
        
        
        val = None if not(param in defaults) else defaults[param]
        
        print("WARNING: ", param, " not found.  Assuming default: ", val)
    
    return val