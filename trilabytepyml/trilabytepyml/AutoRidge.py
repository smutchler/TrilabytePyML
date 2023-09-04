import sys
import json
import pandas as pd
import trilabytepyml.util.Parameters as params 
from multiprocessing import Pool
from trilabytepyml.Ridge import Ridge

def predictOptimal(tdict: dict) -> dict:
    ridge = Ridge()
    return ridge.predict(tdict['frame'], tdict['options'])

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
 
    fdicts = []
    for frame in frames:
        fdict = dict()
        frame = frame[1]
        frame.reset_index(drop=True, inplace=True)
        fdict['frame'] = frame
        fdict['options'] = options
        fdicts.append(fdict)
    
    with Pool() as pool:
        results = pool.map(predictOptimal, fdicts)
        
    for tdict in results:  
        frame = tdict['frame']
        outputFrame = frame if outputFrame is None else outputFrame.append(frame)
    
    return outputFrame

##############################
# Main
##############################
if __name__ == '__main__':
    
    print("AutoRidge - Stacked Data with Role Definition (TRAINING,SCORING)")
    print("-------------------------------")
    print("Required Libraries:")
    print("pip install pandas loess scipy numpy scikit-learn")
    print("-------------------------------")
    print("Usage: python -m src.AutoRidge [json options] [csv source data] [output csv file]")
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