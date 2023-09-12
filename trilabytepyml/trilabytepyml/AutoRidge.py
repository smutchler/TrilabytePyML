import sys
import json
import pandas as pd
import trilabytepyml.util.Parameters as params 
from multiprocessing import Pool
import trilabytepyml.Ridge as ri

def predictOptimal(tdict: dict) -> dict:
    frame = tdict['frame']
    options = tdict['options']
    
    results = dict()
    if 'predictorConstraints' in options:
        # print('Constraining predictors...')       
        constraints = options['predictorConstraints']
        for constraint in constraints:
            # print(f'********** Checking constraint: {constraint}')
            newOptions = constrainOptions(constraint, options)
            # print(f'''Predictors: {newOptions['predictorColumns']}''')
            result = ri.predict(frame.copy(), newOptions)
            frame = result['frame']
            
            # check for X_ERROR; if error then just return result
            if 'X_ERROR' in frame and frame['X_ERROR'][0] is not None:
                # print('Got X_ERROR is None')
                return result
            
            rsqr = getRsqr(frame, constraint)
            # print(f'rsqr: {rsqr}')
            
            results[rsqr] = result
        
        if (max(results.keys() == 0)):
            raise("No valid model was found")
        
        return results[max(results.keys())]
    
    return ri.predict(frame, options)

# remove predictors where constraint = 'exclude'
def constrainOptions(constraint:list, options:dict) -> dict:
    predictors = options['predictorColumns']
    
    if len(constraint) != len(predictors):
        raise('predictorConstraints must be same length as predictorColumns')
        
    newPredictors = []
    for idx in range(len(constraint)):
        constraintValue = constraint[idx]
        predictorColumn = predictors[idx]
        if constraintValue != 'exclude':
            newPredictors.append(predictorColumn)
    
    if len(newPredictors) == 0:
        raise('All predictor columns were excluded by constraints!')
    
    options = options.copy()
    options['predictorColumns'] = newPredictors
    return options

# get Rsqr for a constrained Ridge prediction - return 0 if constraint violated
def getRsqr(frame:pd.DataFrame, constraint:list) -> float:
    # check constraints; if violated return 0
    coefficients = [float(s) for s in frame['X_COEFFICIENTS'][0].split(',')]  
    rsqr = float(frame['X_RSQR'][0])
    # print(f'raw rsqr: {rsqr}')
    
    constraint = [s for s in constraint if s != 'exclude']
    
    for idx in range(len(constraint)):
        constraintValue = constraint[idx]
        coef = coefficients[idx]
        
        if constraintValue == 'negative' and coef > 0:
            # print(f'Negative violated for {idx+1}')
            return 0
        if constraintValue == 'positive' and coef < 0:
            # print(f'Positive violated for {idx+1}')
            return 0
    
    return rsqr


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
    frames = list(frame.groupby(by=params.getParam('splitColumns', options)))
    
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
        outputFrame = frame if outputFrame is None else pd.concat([outputFrame, frame], ignore_index = True)
    
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