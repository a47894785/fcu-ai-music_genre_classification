from email.mime import audio
from wsgiref.validate import InputWrapper
from isort import file
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
from pydub import AudioSegment
import os
import pickle
import random
import operator
import filetype

import math
import numpy as np
from collections import defaultdict

dataset = []
flag = False


def loadDataset(filename):
    with open("mydataset.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break


loadDataset("mydataset.dat")


def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def convertToMp3(fileUrl):
    dst = "convertedMp3.wav"
    audSeg = AudioSegment.from_mp3(fileUrl)
    audSeg.export(dst, format="wav")
    return dst


def checkType(fileName):
    global flag
    audType = filetype.guess(fileName)

    if (audType != None):
        if (audType.mime == "audio/mpeg"):
            inputFile = convertToMp3(fileUrl)
            flag = True
        elif (audType.mime == "audio/x-wav"):
            inputFile = fileUrl
            flag = False

    return inputFile


results = ["", "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


fileUrl = "music1.mp3"
inputFile = checkType(fileUrl)


(rate, sig) = wav.read(inputFile)
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, 0)

pred = nearestClass(getNeighbors(dataset, feature, 5))

print(results[pred])

if (flag):
    os.remove(inputFile)
