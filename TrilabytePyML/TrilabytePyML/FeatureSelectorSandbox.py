from TrilabytePyML.feature.FeatureSelector import FeatureSelector

if __name__ == '__main__':
    
#     pframe = pd.read_csv('c:/temp/iris.csv')
#     selector = FeatureSelector(None)
#     pframe = selector.createSampleData(5000)
#     pframe.to_csv('c:/temp/sample_data.csv')
#     selector.head(100)
    
    selector = FeatureSelector(None)
    pframe = selector.createSampleData(5000)
    selector = FeatureSelector(pframe)
    
    pred_cols = ['VAR_0', 'VAR_1', 'VAR_2', 'VAR_3', 'VAR_4', 'VAR_5', \
                 'VAR_6', 'VAR_7', 'VAR_8', 'VAR_9', 'VAR_10', \
                 'VAR_11', 'VAR_12', 'VAR_13', 'VAR_14', 'VAR_15', \
                 'VAR_16', 'VAR_17', 'VAR_18', 'VAR_19']
    target_col = 'TARGET'
    num_trees = 100
    trial_sizes = [20,15,10,5]
    print(trial_sizes)
    min_mape_loss = 0.1  # 0.1%
    top_vars = selector.selectScaleTarget(target_col, pred_cols, trial_sizes, num_trees, min_mape_loss)
    print(top_vars)
    
