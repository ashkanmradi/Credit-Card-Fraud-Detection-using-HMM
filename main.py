import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sidePackages.HMM import HMM
from sidePackages.Detector import Detector
from sklearn.cluster import DBSCAN



##### default values #####
clusteringMethod = ['kMeans', 'single_link', 'complete_link', 'average_link', 'DBScan']
K = 3
S = 10
thr = 0.9
alpha = 0
clusteringType = clusteringMethod[0]


#### preparing data ####
data = pd.read_csv('data_p1.txt', header = None)
new_data = data.copy()
new_data = data[50:]
new_data = new_data.append([11])


#### Main ####
if(clusteringType == 'kMeans'):
    ##### KMeans Clustering #####
    kmClusters = KMeans(n_clusters=K, random_state=0).fit(data)

    ##### get number of points in each cluster #####
    # print(Counter(kmClusters.labels_))
    # print(Counter(newKmClusters.labels_))
    kmData = np.zeros((len(data),))
    for i in range(len(kmData)):
        kmData[i] = kmClusters.labels_[i]

    newKmData = kmClusters.predict(new_data)

    ##### Initializing the HMM using kMeans Clustering #####
    hmm = HMM(S, K)
    hmm.initializeHMM()
    hmm.train(kmData, 100)

    ##### Detecting fraudulent transaction #####
    detector = Detector(hmm)
    detector.setThreshold(thr)
    alpha = detector.calculateAlpha(kmData, newKmData)
    print("Is it fraud using kmeans? --> ", detector.fraudEvaluation(alpha, newKmData))
    alphaOrd = 0
    alphaOrd = detector.calculateOrdinaryAlpha(kmData, newKmData)
    print("Is it fraud using kmeans? --> ", detector.fraudEvaluation(alphaOrd, newKmData))


elif(clusteringType == 'single_link'):
    ##### Single link clustering #####
    slClusters = AgglomerativeClustering(n_clusters=3, linkage='single').fit(data)

    ##### get number of points in each cluster #####
    # print(Counter(slClusters.labels_))
    # print(Counter(newSlClusters.labels_))
    slData = np.zeros((len(data),))
    for i in range(len(slData)):
        slData[i] = slClusters.labels_[i]

    newSlData = slClusters.fit_predict(new_data)

    ##### Initializing the HMM using Single Link Clustering #####
    hmm = HMM(S, K)
    hmm.initializeHMM()
    hmm.train(slData, 100)

    ##### Detecting fraudulent transaction #####
    detector = Detector(hmm)
    detector.setThreshold(thr)
    alpha = detector.calculateAlpha(slData, newSlData)
    print("Is it fraud using single link? --> ", detector.fraudEvaluation(alpha, newSlData))

    alphaOrd = 0
    alphaOrd = detector.calculateOrdinaryAlpha(slData, newSlData)
    print("Is it fraud using single Link? --> ", detector.fraudEvaluation(alphaOrd, newSlData))
    #

elif(clusteringType == 'complete_link'):
    ##### Complete link clustering #####
    clClusters = AgglomerativeClustering(n_clusters=3, linkage='complete').fit(data)
    newClClusters = AgglomerativeClustering(n_clusters=3, linkage='complete').fit(new_data)
    ##### get number of points in each cluster #####
    # print(Counter(clClusters.labels_))
    # print(Counter(newClClusters.labels_))
    clData = np.zeros((len(data),))
    for i in range(len(clData)):
        clData[i] = clClusters.labels_[i]

    newClData = clClusters.fit_predict(new_data)

    ##### Initializing the HMM using Complete Link Clustering #####
    hmm = HMM(S, K)
    hmm.initializeHMM()
    hmm.train(clData, 100)

    ##### Detecting fraudulent transaction #####
    detector = Detector(hmm)
    detector.setThreshold(thr)
    alpha = detector.calculateAlpha(clData, newClData)
    print("Is it fraud using complete link? --> ", detector.fraudEvaluation(alpha, newClData))

    alphaOrd = 0
    alphaOrd = detector.calculateOrdinaryAlpha(clData, newClData)
    print("Is it fraud using complete link? --> ", detector.fraudEvaluation(alphaOrd, newClData))

elif(clusteringType == 'average_link'):
    ##### Average link clustering #####
    alClusters = AgglomerativeClustering(n_clusters=3, linkage='average').fit(data)
    newAlClusters = AgglomerativeClustering(n_clusters=3, linkage='average').fit(new_data)
    ##### get number of points in each cluster #####
    # print(Counter(alClusters.labels_))
    # print(Counter(newAlClusters.labels_))
    alData = np.zeros((len(data),))
    for i in range(len(alData)):
        alData[i] = alClusters.labels_[i]

    newAlData = alClusters.fit_predict(new_data)

    ##### Initializing the HMM using Average Link Clustering #####
    hmm = HMM(S, K)
    hmm.initializeHMM()
    hmm.train(alData, 100)
    ##### Detecting fraudulent transaction #####
    detector = Detector(hmm)
    detector.setThreshold(thr)
    alpha = detector.calculateAlpha(alData, newAlData)
    print("Is it fraud using average link? --> ", detector.fraudEvaluation(alpha, newAlData))

    alphaOrd = 0
    alphaOrd = detector.calculateOrdinaryAlpha(alData, newAlData)
    print("Is it fraud using average link? --> ", detector.fraudEvaluation(alphaOrd, newAlData))


elif(clusteringType == 'DBScan'):
    ##### DBScan clustering #####
    DBSClusters = DBSCAN().fit(data)
    newDBSClusters = DBSCAN().fit(new_data)
    ##### get number of points in each cluster #####
    # print(Counter(DBSClusters.labels_))
    # print(Counter(newDBSClusters.labels_))
    DBSData = np.zeros((len(data),))
    for i in range(len(DBSData)):
        DBSData[i] = DBSClusters.labels_[i]

    newDBSData = DBSClusters.fit_predict(new_data)
    K = len(np.unique(DBSClusters.labels_))
    ##### Initializing the HMM using DBScan Clustering #####
    hmm = HMM(S, K)
    hmm.initializeHMM()
    hmm.train(DBSData, 100)

    ##### Detecting fraudulent transaction #####
    detector = Detector(hmm)
    detector.setThreshold(thr)
    alpha = detector.calculateAlpha(DBSData, newDBSData)
    print("Is it fraud using DBScan? --> ", detector.fraudEvaluation(alpha, newDBSData))

    alphaOrd = 0
    alphaOrd = detector.calculateOrdinaryAlpha(DBSData, newDBSData)
    print("Is it fraud using DBScan? --> ", detector.fraudEvaluation(alphaOrd, newDBSData))













