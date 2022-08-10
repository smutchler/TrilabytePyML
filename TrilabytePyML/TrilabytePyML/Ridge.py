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
import TrilabytePyML.util.Parameters as params 
import random

def buildSampleoptionsJSONFile(jsonFileName: str) -> None:
    """
    This function creates a JSON file storing some of the options information
    for use with the iris_ridge data in the samples folder that comes with
    the module. It should be saved to the working directory under the name
    passed as the jsonFileName parameter.

    Parameters
    ----------
    jsonFileName : str
        Desired name of JSON file

    Returns
    -------
    None
        No output

    """
    options = dict()
    options['splitColumns'] = ['Split']
    options['predictorColumns'] = ['Sepal.Width', 'Sepal.Length', 'Petal.Width']
    options['roleColumn'] = 'Role'
    options['targetColumn'] = 'Petal.Length' 
    options['ridgeAlpha'] = 1.0
    
    # print(json.dumps(options))
    
    with open(jsonFileName, 'w') as fp:
        json.dump(options, fp)

        
def detectOutliers(frame: pd.DataFrame, options: dict):
    '''
    Find target values that are outsize X standard deviations of the mean
    add column X_OUTLIER where 1 = outlier; 0 = non-outlier
    '''
    # stdev = frame[options['targetColumn']].std()
    # avg = frame[options['targetColumn']].mean()
    
    mult = params.getParam('outlierStdevMultiplier', options)
    
    #identifies outliers based on the number of standard deviations
    frame['X_OUTLIER'] = 0
    for index, row in frame.iterrows():
        val = abs(frame[options['targetColumn']][index])
        data = frame[options['targetColumn']].copy()
        data = data.drop(index)
        
        # check for sufficient data
        if (len(data) >= 2):
            stdev = data.std()
            avg = data.mean()
            
            if val > avg + mult * stdev:
                    frame['X_OUTLIER'][index] = 1
            else:
                    frame['X_OUTLIER'][index] = 0  
        else:
            frame['X_OUTLIER'][index] = 0
    
    return(frame)

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
                
    model = Ridge(alpha=options['ridgeAlpha'])
    model.fit(x, y)
    
    xscore = frame[options['predictorColumns']]
    yhat = model.predict(xscore)
    
    frame['X_PREDICTED'] = yhat

    fdict['frame'] = frame
    return(fdict)

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
    frames = list(frame.groupby(by=options['splitColumns']))
    
    outputFrame = None
 
    for frame in frames:
        frame = frame[1]
        frame.reset_index(drop=True, inplace=True)
        
        fdict = predict(frame, options)
        frame = fdict['frame']
         
        outputFrame = frame if outputFrame is None else outputFrame.append(frame)
    
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
    print(json.dumps(options,indent=2), '\n')

    frame = pd.read_csv(fileName)
    
    outputFrame = splitIntoFramesAndPredict(frame, options)
     
    outputFrame.to_csv(outputFileName, index=False)
     
    print("Output file: ", outputFileName)
     
    print("Predictions complete...")