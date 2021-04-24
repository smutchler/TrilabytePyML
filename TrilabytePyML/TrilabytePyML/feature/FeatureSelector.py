#
#    http://trilabyte.com
#    Trilabyte Python Machine Learning
#    Copyright (c) 2020 - Trilabyte
#    Author: Scott Mutchler
#    Contact: smutchler@trilabyte.com
#
#
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from TrilabytePyML.stats.Statistics import calcMAPE
from random import random
import numpy as np

class FeatureSelector:
    
    pframe = None
 
    def __init__(self, pframe: pd.DataFrame):
        self.pframe = pframe 
    
    def head(self, numRows: int):
        print(self.pframe.head(numRows))
    
    def selectScaleTarget(self, target_col, pred_cols, trial_sizes, n_trees, min_mape_loss):
        min_mape = 1E10
        for trial_size in trial_sizes:

            x_train, x_test, y_train, y_test = train_test_split(self.pframe[pred_cols], self.pframe[target_col], test_size=0.3, random_state=3093299176)
            
            model = RandomForestRegressor(n_estimators=n_trees)
            model.fit(x_train, y_train)
            
            imp_frame = pd.DataFrame()
            imp_frame['Var'] = pred_cols 
            imp_frame['Importance'] = model.feature_importances_
            imp_frame.sort_values(by=['Importance'], ascending=False, inplace=True)
            imp_frame.reset_index(inplace=True, drop=True)
            top_vars = imp_frame['Var'][range(trial_size)]
#             print("top variables:", top_vars)
#              
            predictions = model.predict(x_test)
            mape = calcMAPE(pd.Series(predictions), pd.Series(y_test))
            
            if (mape - min_mape) < min_mape_loss:
                pred_cols = top_vars
                if mape < min_mape:
                    min_mape = mape
            else:
                return pred_cols
        
        print("trial size:", trial_size, "mape:", mape, "min_mape:", min_mape, "mape diff:", (mape - min_mape), "min_mape_loss:",min_mape_loss)
        
        return pred_cols
    
    def createSampleData(self, n_rows: int):
        n_cols = 20
        x_coef = [] 
        for i in range(n_cols):
            if i == 0:
                x_coef.append(random())
            else:
                x_coef.append(x_coef[i - 1] * 1.25)
        
#         print(x_coef)
        
        pframe = pd.DataFrame()
        
        for col in range(n_cols):
            col_vals = []
            for row in range(n_rows):
                col_vals.append(random())
            pframe['VAR_' + str(col)] = col_vals
        
        pframe['TARGET'] = pframe.apply(lambda row : np.dot(x_coef, row) + random() * 5.0, axis=1)
        
#         print(pframe.head(100))
        return(pframe)
            
