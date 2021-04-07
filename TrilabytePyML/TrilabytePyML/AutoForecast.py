#
#    http://trilabyte.com
#    Trilabyte Python Machine Learning
#    Copyright (c) 2020 - Trilabyte
#    Author: Scott Mutchler
#    Contact: smutchler@trilabyte.com
#
# more comments

import json 
import sys
import pandas as pd 
from TrilabytePyML.Forecast import Forecast
import TrilabytePyML.util.Parameters as params
import traceback 
from statistics import median
from TrilabytePyML.stats.Statistics import calcMAPE

def findMAPE(frame: pd.DataFrame, options: dict, seasonality: str) -> float:
    """
    This function takes the data through the "frame" parameter, takes the 
    instructions as to how to read the data through the 'options' dictionary
    parameter, and then takes the seasonality that should be applied to the
    instance of MLR forecast that forms part of this function.

    Parameters
    ----------
    frame : pd.DataFrame
        Data to use in forecast
    options : dict
        Options dictionary specifying how to read 'frame'
    seasonality : str
        'Additive' or 'Multiplicative'

    Returns
    -------
    dict
        Returns the MAPE on the results of the forecastMLR() forecast and the 
        actuals column

    """
    options = options.copy()
    options['seasonality'] = seasonality
    model = Forecast()
    return  model.forecastMLR(frame.copy(), options)['MAPE']


def findOptimalSeasonality(frame: pd.DataFrame, options: dict) -> str:
    """
    Compares the MAPE as calculated in findMAPE() to determine whether
    'additive', 'multiplicative', or 'none' is the best seasonality
    qualifier to apply

    Parameters
    ----------
    frame : pd.DataFrame
        Data to use in forecast
    options : dict
        Instructions as to how to read the data

    Returns
    -------
    str
        Will be "None", "Additive", or "Multiplicative"

    """
    nonNullRowCount = frame[params.getParam('targetColumn', options)].count()
    periodicity = params.getParam('periodicity', options)
    
    if nonNullRowCount < periodicity:
        return "None"
    
    noSeasonalityMAPE = findMAPE(frame, options.copy(), 'None')
    additiveMAPE = findMAPE(frame, options.copy(), 'Additive')
    multiplicativeMAPE = findMAPE(frame, options.copy(), 'Multiplicative')
    
    minMAPE = min(noSeasonalityMAPE, additiveMAPE, multiplicativeMAPE)
    
    # in case of ties this returns a hierarchy of simple to complex
    if (noSeasonalityMAPE == minMAPE):
        return "None"
    elif (additiveMAPE == minMAPE):
        return "Additive"
    else:
        return "Multiplicative"


def splitFramesAndForecast(frame: pd.DataFrame, options: dict) -> pd.DataFrame:
    """
    Automates forecasting for each subframe defined by the sortColumns options
    parameter. Will return the forecast using the method with the best MAPE for
    each subframe. Will also print the MAPEs of each method for each forecast
    made.

    Parameters
    ----------
    frame : pd.DataFrame
        Data needed for forecast
    options : dict
        Instructions on how to read 'frame'

    Returns
    -------
    outputFrame : pd.DataFrame
        Same as original 'frame' but with all the columns associated with a
        forecast added.

    """
    #creates a list of frames, each of which will correspond to a different
    #forecast
    frame.sort_values(by=params.getParam('sortColumns', options), ascending=True, inplace=True)
    
    frames = list(frame.groupby(by=params.getParam('splitColumns', options)))
    
    outputFrame = None

    for frame in frames:
            frame = frame[1]
            frame.reset_index(drop=True, inplace=True)
            
            method = params.getParam('method', options)
            
            #specifies actions if the forecast method is set to "Auto" in 
            #the options dictionary
            if method == 'Auto':
                opts = options.copy()
                opts['method'] = 'ARIMA'
                arimaFrame = forecastSingleFrame(frame.copy(), opts)
                arimaMAPE = 1E6 if 'X_MAPE' not in arimaFrame else arimaFrame['X_MAPE'][0]
                
                opts = options.copy()
                opts['method'] = 'Prophet'
                prophetFrame = forecastSingleFrame(frame.copy(), opts)
                prophetMAPE = 1E6 if 'X_MAPE' not in prophetFrame else prophetFrame['X_MAPE'][0]
                
                opts = options.copy()
                opts['method'] = 'MLR'
                mlrFrame = forecastSingleFrame(frame.copy(), opts)
                mlrMAPE = 1E6 if 'X_MAPE' not in mlrFrame else mlrFrame['X_MAPE'][0]
                
                if 'X_FORECAST' in mlrFrame and 'X_FORECAST' in prophetFrame and 'X_FORECAST' in arimaFrame:
                    ensembleFrame = mlrFrame.copy() 
                    
                    # we calculate MAPE using original data column
                    targetColumn = params.getParam('targetColumn', options)
                    if (targetColumn.startswith('X_')):
                        targetColumn = targetColumn[2:]
                    
                    # split the data into past/future based on null in target column 
                    numHoldoutRows = params.getParam('numHoldoutRows', options)
                    lastNonNullIdx = Forecast().lastNonNullIndex(ensembleFrame[targetColumn])
                    lastNonNullIdx = lastNonNullIdx - numHoldoutRows
        
                    if (numHoldoutRows > 0):
                        evalIdx = list(map(lambda x: x > lastNonNullIdx and x <= (lastNonNullIdx + numHoldoutRows), ensembleFrame['X_INDEX']))
                    else:
                        evalIdx = ensembleFrame['X_INDEX'] <= lastNonNullIdx
                    
                    ensembleFrame['X_FORECAST'] = list(map(lambda x, y , z: median([x, y, z]), mlrFrame['X_FORECAST'], arimaFrame['X_FORECAST'], prophetFrame['X_FORECAST']))
                    ensembleFrame['X_LPI'] = list(map(lambda x, y , z: median([x, y, z]), mlrFrame['X_LPI'], arimaFrame['X_LPI'], prophetFrame['X_LPI']))
                    ensembleFrame['X_UPI'] = list(map(lambda x, y , z: median([x, y, z]), mlrFrame['X_UPI'], arimaFrame['X_UPI'], prophetFrame['X_UPI']))
                    
                    evalFrame = ensembleFrame[evalIdx]
                    try:
                        ensembleMAPE = calcMAPE(evalFrame['X_FORECAST'], evalFrame[targetColumn])
                        ensembleFrame['X_MAPE'] = ensembleMAPE
                        for index, row in ensembleFrame.iterrows():
                            ensembleFrame['X_APE'][index] = (abs(row['X_FORECAST'] - row[targetColumn]) / row[targetColumn] * 100.0) if row[targetColumn] != 0 else None
                    except:
                        # this may be needed if all forecasts frame and MAPE, APE cannot be calculated
                        if (not('X_MAPE' in ensembleFrame)):
                            ensembleFrame['X_MAPE'] = 1E6
                        if (not('X_APE' in ensembleFrame)):
                            ensembleFrame['X_APE'] = 1E6
                        
                    mapes = [mlrMAPE, arimaMAPE, prophetMAPE, ensembleMAPE]
                else:
                    mapes = [mlrMAPE, arimaMAPE, prophetMAPE]
                
                print("Auto MAPEs (MLR, ARIMA, Prophet, Ensemble): ", mapes)
                
                minMAPE = min(mapes)
                
                if (mlrMAPE <= minMAPE):
                    frame = mlrFrame
                    frame['X_METHOD'] = 'MLR'
                elif (prophetMAPE <= minMAPE):
                    frame = prophetFrame
                    frame['X_METHOD'] = 'Prophet'
                elif (arimaMAPE <= minMAPE):
                    frame = arimaFrame
                    frame['X_METHOD'] = 'ARIMA'      
                else:
                    frame = ensembleFrame
                    frame['X_METHOD'] = 'Ensemble'                         
                
            else:
                frame = forecastSingleFrame(frame, options.copy())
            
            outputFrame = frame if outputFrame is None else outputFrame.append(frame, ignore_index=True)
    
    return outputFrame

def forecastSingleFrame(frame: pd.DataFrame, options: dict) -> pd.DataFrame:
    """
    Basically the equivalent of splitFramesAndForecast() but for a single frame

    Parameters
    ----------
    frame : pd.DataFrame
        Data needed to forecast
    options : dict
        Instructions as to how to handle 'frame'

    Returns
    -------
    frame : pd.DataFrame
        The original 'frame' parameter but with additional columns for the
        forecast.

    """
    try:
        method = params.getParam('method', options)
        currentOptions = options.copy()
        
        model = Forecast()
                    
        if (method == 'MLR'):
            if params.getParam('seasonality', options) == 'Auto':
                currentOptions['seasonality'] = findOptimalSeasonality(frame.copy(), options.copy())
            
            fdict = model.forecastMLR(frame, currentOptions.copy())
        elif method.lower() == 'Prophet'.lower():
            fdict = model.forecastProphet(frame, options.copy())
        else:
            fdict = model.forecastARIMA(frame, currentOptions.copy())
        
        frame = fdict['frame']
        frame['X_ERROR'] = None 
        frame['X_METHOD'] = method
    
    except Exception as e:
        ed = str(traceback.format_exc()).replace('\n', ' ')
        frame['X_ERROR'] = e
        frame['X_METHOD'] = method
    
    return frame
        

##############################
# Main
##############################
if __name__ == '__main__':
    
    print("AutoForecast")
    print("-------------------------------")
    print("*** You must use Anaconda for Facebook Prophet")
    print("")
    print("Required Librarires:")
    print("pip install pandas loess scipy numpy scikit-learn pmdarima")
    print("conda install -c anaconda ephem")
    print("conda install -c conda-forge pystan fbprophet")
    print("-------------------------------")
    print("Usage: python -m src.AutoForecast [json forecastMLR options] [csv source data] [output csv file]")
    print("-------------------------------")
  
    pd.options.mode.chained_assignment = None  # default='warn'
  
    DEBUG = False 
  
    if DEBUG:
        fileName = 'c:/temp/retail_unit_demand.csv'
        jsonFileName = 'c:/temp/retail_unit_demand_options.json'
        outputFileName = 'c:/temp/retail_unit_demand_forecast.csv'
    else:
        if (len(sys.argv) < 3):
            print("Error: Insufficient arguments")
            sys.exit(-1)
                    
        jsonFileName = sys.argv[1]
        fileName = sys.argv[2]
        outputFileName = sys.argv[3]
    
    with open(jsonFileName, 'r') as fp:
        options = json.load(fp)
    
    print('Options:') 
    print(json.dumps(options, indent=2), '\n')
    
    frame = pd.read_csv(fileName)
        
    outputFrame = splitFramesAndForecast(frame, options)
    
    outputFrame.to_csv(outputFileName, index=False)
    
    print("Output file: ", outputFileName)
    
    print("Forecast(s) complete...")
