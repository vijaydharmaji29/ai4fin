import pandas as pd

# Load the first file
df1 = pd.read_csv('rmse_results.csv')
newDf = {'Commodity':[], 'GARCH':[], 'GJR-GARCH':[], 'EWMA':[], 'MEM':[], 'P-WEV':[]}

i = 0

while(i < len(df1)):
    newDf['Commodity'].append(df1.iloc[i]['File'].split('_')[0])
    newDf['GARCH'].append(df1.iloc[i]['RMSE'])
    newDf['GJR-GARCH'].append(df1.iloc[i+1]['RMSE'])
    newDf['EWMA'].append(df1.iloc[i+2]['RMSE'])
    newDf['MEM'].append(df1.iloc[i+3]['RMSE'])
    newDf['P-WEV'].append(df1.iloc[i+4]['RMSE'])
    i += 5

newDf = pd.DataFrame(newDf)
newDf.to_csv('../formatted_volatility_results/rmse_volatility_results.csv', index=False)
