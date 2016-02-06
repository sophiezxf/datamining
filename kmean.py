__author__ = 'Xiaofei Zhang'

import numpy as np
from math import *
import random
import csv
import sys

def csvReader(filename):

    f = open(filename, 'rU')
    reader = csv.reader(f, delimiter=',', skipinitialspace=True)

    headers = reader.next()
    # find attributes
    newHeaders = ['latitude', 'longitude', 'reviewCount', 'checkins']

    oldcolumn = {}

    for h in headers:
        oldcolumn[h] = []

    for row in reader:
        for h, v in zip(headers, row):
            oldcolumn[h].append(v)

    #print (newHeaders)

    column = {}

    for h in newHeaders:
        column[h] = []

    rowNumber = 0
    for h in newHeaders:
        column[h] = oldcolumn.get(h)
        rowNumber = rowNumber + 1

    return column


def vc(train):
    # the purpose of this function si to construct the traindata
    trainSize = len(train['latitude'])
    vectorCandidates = []

    ## append the training data value
    for i in range(trainSize):
        vectorCandidates.append([train['latitude'][i],
                      train['longitude'][i],
                      train['reviewCount'][i],
                      train['checkins'][i]])

    return vectorCandidates


def DistEuc (vector1, vector2):
    # find the disEuc distance

    return sqrt(sum((vector1 - vector2) ** 2))

def iniCent (dataSet, k):
    '''construct a set containing k centroid
    return centroid'''
    # tuple of dataSet dimension
    ns, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        #random.seed(100)
        index = int(random.uniform(0, ns))
        centroids[i, :] = dataSet[index, :]
    return centroids

# k-means cluster
def kmeans(dataSet, k):

    ns = dataSet.shape[0]


    groupsOfClu = np.mat(np.zeros((ns, 2)))
    ifClusterMoved = True

    # initiate the centttriods
    centroids = iniCent (dataSet, k)

    while ifClusterMoved:
        ifClusterMoved = False


        for i in range(ns):
            distMin = 1.0e31
            minIndex = 0
            # check the smallest distance !!!!
            for j in range(k):

                dist = DistEuc(centroids[j, :], dataSet[i, :])
                if dist < distMin:
                    distMin = dist
                    minIndex = j


            if groupsOfClu[i, 0] != minIndex:
                ifClusterMoved = True
                ## give min mark and min distance
                groupsOfClu[i, :] = minIndex, distMin ** 2

        # recursive
        for xxx in range(k):
            pointsInCluster = dataSet[np.nonzero(groupsOfClu[:,0].A == xxx)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)

    # print "groupsOfClu:", groupsOfClu

    # get the sum of all within cluster distance
    # wcsum = 0
    #print type(groupsOfClu)
    #print groupsOfClu[:,1]
    # i in groupsOfClu:


    # wcsum = sum(groupsOfClu[:,1])
    # print "WC-SSE=", wcsum[0,0]

    return centroids, groupsOfClu

# main function
def main():
    if len(sys.argv) != 3:
        print "input wrong \n" \
              "correct input formt \n" \
              "nbc.py train-set.csv K !!!"
        exit()

    train = csvReader(sys.argv[1])
    # convert K into int type
    K = int(sys.argv[2])


    #print (K)
    #print (train['latitude'][0:10])

    ## construct the candidate vector
    vectorCandidates = vc(train)

    ## convert dataset format
    dataset_tem = np.array(vectorCandidates)
    dataset = dataset_tem.astype(np.float)

    ## test distance
    # print "dataset:",dataset
    # print(DistEuc(dataset[0], dataset[1]))

    ## test iniCent
    temc = iniCent (dataset, K)

    #print (type(temc))
    # print (temc)
    centroids, groupsOfClu = kmeans(dataset, K)

    wcsum = 0
    wcsum = sum(groupsOfClu[:,1])

    ## print the result
    print "WC-SSE =", wcsum[0,0]


    for i in range(0,len(centroids)):
        print "Centroid%s"%(i+1),"=",centroids[i]
    #print (centroids)

if __name__ == "__main__":
    main()

