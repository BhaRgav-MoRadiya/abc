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
dataset = read_csv('ToBeConverted-2.csv')
# manually specify column names
dataset.columns = ['EdgeId', 'LearningEdge', 'Congestion', 'Density', 'GpsProbe', 'Speed']

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
    
        
import pandas as pd
dataset.drop(['EdgeId','LearningEdge','Congestion'],axis=1,inplace=True)
df = pd.DataFrame([result1,result2,result3,list(dataset.Density),list(dataset.GpsProbe),list(dataset.Speed)],dtype=object)
df=pd.DataFrame.transpose(df)
df.columns=['EdgeId','LearningEdge','Congestion','Density','GpsProbe','Speed']
df.set_index('EdgeId', inplace=True)
df.to_csv('tex-2.csv')
