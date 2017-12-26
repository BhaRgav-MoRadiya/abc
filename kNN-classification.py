# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import pandas as pd
import numpy as np
def scale(dataset):
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    df = pd.DataFrame(np.array(dataset).reshape(37891,9))
    df = df.values
    #[0=edgeId,1=congestion level,2=occupancy,3=gps,4=speed,5=startTime,6=endTime,7=totalCount,8=exitCount]
    scale.fit(df[:,[1,2,3,4,7,8]])
    df[:,[1,2,3,4,7,8]]= scale.transform(df[:,[1,2,3,4,7,8]])
    return list(df)
    
    
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset=scale(dataset)
        for x in range(len(dataset)-1):
	        for y in range(8):
	            dataset[x][y+1] = float(dataset[x][y+1])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in [1,2,3,4,7,8]:
		distance += abs((instance1[x] - instance2[x]))
	return (distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][0]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][0] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('Full-data1.data', split, trainingSet, testSet)
	print ('Train set: ' + repr(len(trainingSet)))
	print ('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][0]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()