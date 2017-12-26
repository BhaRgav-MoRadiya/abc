from pandas import read_csv
def mapValue(x):
    if x=="A":
        return 0
    elif x=="B":
        return 1
    elif x=="BC":
        return 2
    elif x=="C":
        return 3
    elif x=="CD":
        return 4
    elif x=="D":
        return 5
    elif x=="DE":
        return 6
    elif x=="E":
        return 7
    elif x=="EF":
        return 8
    else:
        return 9

# load data
dataset = read_csv('ToBeConverted.csv')
#dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['EdgeId', 'LearningEdge', 'Congestion', 'Desnsity', 'StartTime', 'EndTime', 'GpsProbe', 'Speed']
#dataset.index.name = 'EdgeId'
# mark all NA values with 0
#dataset['Converted'].fillna(0, inplace=True)
# drop the first 24 hours
#dataset = dataset[24:]
from builtins import int
result1=list(dataset.EdgeId)
for i in range(len(result1)):
    result1[i]=int(''.join(filter(str.isdigit, result1[i])))
    i+=1
result2=list(dataset.LearningEdge)
for i in range(len(result2)):
    result2[i]=int(''.join(filter(str.isdigit, result2[i])))
    i+=1
result3=list(dataset.Congestion)
for i in range(len(result3)):
    result3[i]=mapValue(result3[i])
    i+=1
    
    
    
#df=pd.DataFrame([result2],dtype=object) 
#df.to_csv('tex.csv')
# save to file
#dataset.to_csv('Converted.csv')

#dataset[1]=result2
#import numpy as np
import pandas as pd
dataset.drop(['EdgeId','LearningEdge','Congestion'],axis=1,inplace=True)
#result3= pd.to_numeric(np.column_stack((result1,result2)))
#df = pd.DataFrame(np.column_stack((np.column_stack((result1,result2)),dataset)))
df = pd.DataFrame([result1,result2,result3,list(dataset.Desnsity),list(dataset.StartTime),list(dataset.EndTime),list(dataset.GpsProbe),list(dataset.Speed)],dtype=object)
df=pd.DataFrame.transpose(df)
df.columns=['EdgeId','LearningEdge','Congestion','Density','StartTime','EndTime','GpsProbe','Speed']
df.set_index('EdgeId', inplace=True)
df.to_csv('Converted.csv')
#df=pd.DataFrame('EdgeId': result1,'LearningEdge': result2,'Congestion': result3)