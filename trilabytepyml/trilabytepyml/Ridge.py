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
from sklearn.linear_model import Ridge
import trilabytepyml.util.Parameters as params 
import random
from multiprocessing import Pool
import warnings

#
# Detect outliers using stdev from mean (after removing seasonality) - if it's an outlier put
# a 1 in the X_OUTLIER column otherwise 0
#
def detectOutliers(frame: pd.DataFrame, options: dict):
    '''
    Find target values that are outsize X standard deviations of the mean
    add column X_OUTLIER where 1 = outlier; 0 = non-outlier
    '''
    # stdev = frame[options['targetColumn']].std()
    # avg = frame[options['targetColumn']].mean()
    
    mult = params.getParam('outlierStdevMultiplier', options)
    
    # identifies outliers based on the number of standard deviations
    frame['X_OUTLIER'] = 0
    for index, row in frame.iterrows():
        val = abs(frame[options['targetColumn']][index])
        data = frame[options['targetColumn']].copy()
        data = data.drop(index)
        
        # check for sufficient data
        if (len(data) >= 2):
            stdev = data.std()
            avg = data.mean()
            
            # fix for near zero stdev
            stdev = max(abs(avg) ** 0.5, stdev)
            
            if val > avg + mult * stdev:
                    frame['X_OUTLIER'][index] = 1
            else:
                    frame['X_OUTLIER'][index] = 0  
        else:
            frame['X_OUTLIER'][index] = 0
    
    return(frame)


def calcContributions(x, model, options):
    '''
    Calculate the regression coeff * predictor value for all columns (to calc contribution
    of each variable on the final regression output)
    '''
    try:
        vals = x[options['predictorColumns']] * model.coef_
        return ','.join(map(str, vals))
    except:
        return None


def predictThreadWrapper(tdict: dict) -> dict:
    return predict(tdict['frame'], tdict['options'])


def predict(frame: pd.DataFrame, options: dict) -> dict:
    """
    The function takes as an argument the "frame" parameter, which is a 
    pandas dataframe with the data involved in the forecast, and the "options"
    parameter, which is a dictionary with at least the roleColumn, 
    predictorColumns, targetColumn, and ridgeAlpha keys defined. The function
    takes the training subset of the frame parameter and trains a ridge
    regression on it. It then applies the ridge regression to the entire
    dataset and returns fdict. fdict['frame'] will return a pandas dataframe 
    that is identical to the original "frame" parameter in every way except 
    for the inclusion of a X_PREDICTED column, which will be the ridge
    model results. Utilized in the splitIntoFramesAndPredict function.

    Parameters
    ----------
    frame : pd.Dataframe
        pandas dataframe with info to be forecast
    options : dict
        dictionary of options and parameters for the module

    Returns
    -------
    dict
        Has a single key, 'frame', which contains the original dataframe
        passed through the "frame" parameter, plus a column specifying the
        model predictions

    """
    pd.options.mode.chained_assignment = None
    
    fdict = dict()
    
    # handle hold-out fraction
    holdoutFraction = params.getParam('holdoutFraction', options)
    frame['X_HOLDOUT'] = frame.apply(lambda x: 1 if (x[options['roleColumn']] == 'TRAINING') & (random.random() < holdoutFraction) else 0, axis=1)
        
    if params.getParam('autoDetectOutliers', options):
        frame = detectOutliers(frame, options)
    else:
        frame['X_OUTLIER'] = 0
        
    trainFrame = frame.loc[(frame[options['roleColumn']] == 'TRAINING') & (frame['X_OUTLIER'] == 0) & (frame['X_HOLDOUT'] == 0)]
    
    x = trainFrame[options['predictorColumns']]
    y = trainFrame[options['targetColumn']]
            
    positiveCoefficents = params.getParam('forcePositiveCoefficients', options)   
    alpha = params.getParam('ridgeAlpha', options)     
                
    model = Ridge(alpha=alpha, positive=positiveCoefficents)
    model.fit(x, y)
    
    xscore = frame[options['predictorColumns']]
    yhat = model.predict(xscore)
        
    frame['X_RSQR'] = model.score(x, y)
    frame['X_INTERCEPT'] = model.intercept_
    frame['X_PREDICTORS'] = ','.join(options['predictorColumns'])
    frame['X_COEFFICIENTS'] = ','.join(map(str, model.coef_))
    frame['X_VAR_CONTRIBUTIONS'] = frame.apply(lambda x: calcContributions(x, model, options), axis=1)
    frame['X_PREDICTED'] = yhat

    fdict['frame'] = frame
    return(fdict)


#
# DEPRECATED: Use AutoRidge.splitIntoFramesAndPredict
#
def splitIntoFramesAndPredict(frame: pd.DataFrame, options: dict) -> pd.DataFrame:
    """
    This function expands on the functionality of the predict() function
    and allows for several predictions to be run on different groupings
    of the original "frame" parameter. These groupings are determined by
    the only additional "options" key that must be specified for this function,
    ['splitColumns']. 
    
    NOTE: Future versions may want to consider not requiring the setting of
    a ['splitColumns'] and defaulting to predict() if no value is defined for
    that key in options

    Parameters
    ----------
    frame : pd.Dataframe
        pandas dataframe with information to be forecast
    options : dict
        provides parameters for the forecast

    Returns
    -------
    outputFrame : TYPE
        Returns multiple forecasts

    """
    warnings.warn("Deprecated: Use AutoRidge.splitIntoFramesAndPredict")
    
    pd.options.mode.chained_assignment = None
    
    frames = list(frame.groupby(by=options['splitColumns']))
    
    outputFrame = None
 
    fdicts = []
    for frame in frames:
        fdict = dict()
        frame = frame[1]
        frame.reset_index(drop=True, inplace=True)
        fdict['frame'] = frame
        fdict['options'] = options
        fdicts.append(fdict)
    
    with Pool() as pool:
        results = pool.map(predictThreadWrapper, fdicts)
        
    for tdict in results: 
        frame = tdict['frame']
        outputFrame = frame if outputFrame is None else pd.concat([outputFrame, frame], ignore_index=True)
    
    return outputFrame


##############################
# Main
##############################
if __name__ == '__main__':
    
    print("Ridge - Stacked Data with Role Definition (TRAINING,SCORING)")
    print("-------------------------------")
    print("Required Libraries:")
    print("pip install pandas loess scipy numpy scikit-learn")
    print("-------------------------------")
    print("Usage: python -m src.Ridge [json options] [csv source data] [output csv file]")
    print("-------------------------------")

    DEBUG = True
    
    if DEBUG:
        fileName = 'c:/temp/iris_with_role_and_split.csv'
        outputFileName = 'c:/temp/iris_ridge.csv'
        jsonFileName = 'c:/temp/iris_ridge.json'
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
    
    outputFrame = splitIntoFramesAndPredict(frame, options)
     
    outputFrame.to_csv(outputFileName, index=False)
     
    print("Output file: ", outputFileName)
     
    print("Predictions complete...")
