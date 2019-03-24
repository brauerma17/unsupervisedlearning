import csv
from sklearn import tree
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from math import ceil
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


fn = "phishing_dataset.csv"
writeFn = fn[:-4] + "_pca_EM_output.csv"
print(writeFn)

covs = ['full','tied','diag','spherical']
n_inits = [5,10,15,20]
max_iters = [100,200,300,500]
norms = [True,False]
#fig = plot.figure(1)
#ax = Axes3D(fig)

def runEverything():
    data,x,y,normx,normy = preProcessData(fn)
    results = []
    for n_comp in range(1,len(x[0])):
        for cov in covs:
            for norm in norms:
                err = classify(x,y,normx,normy,n_comp,cov,norm)
                print(n_comp,cov,norm,":",err)
                results.append([n_comp,cov,norm,err])

    file = open(writeFn,"w",newline="")
    csvW = csv.writer(file)
    csvW.writerow(["n_comp","cov","norm","error"])
    csvW.writerows(results)
    print("Open your file")
    #fig.show()

    fig2 = plot.figure(2)
    ax2 = Axes3D(fig2)
    labels = np.array(y)
    X = np.array(x)
    ax2.scatter(X[:,0],X[:,1],X[:,2],c=labels.astype(np.float),edgecolor='k')
    fig2.savefig(writeFn +'2.png' )
    fig2.show(True)


def norm(vec):
    dist = max(vec) - min(vec)
    new = []
    for item in vec:
        if dist != 0:
            new.append(item/dist)
        else:
            new.append(0)
    return new

def preProcessData(fn):
    data = []
    f = open(fn)

    csvReader = csv.reader(f,delimiter=",")
    for row in csvReader:
        data.append(row)

    data = data[1:]
    i = 0
    for col0 in data[0]:
        try:
            float(col0)
            for row in data:
                row[i] = float(row[i])
        except:
            col = []
            for row in data:
                col.append(row[i])
            le = preprocessing.LabelEncoder()
            le.fit(col)
            a = le.transform(col)
            newCol = np.array(a).tolist()
            #print(newCol)

            for row in data:
                #print(i)
                row[i] = newCol[i]
        i = i + 1

    x = []
    y = []
    for row in data:
        x.append(row[:-1])
        y.append(row[-1])

    #print(x[5:10])

    ## normalize every vector ###
    arrx = np.array(x)
    for i in range(len(x[0])):
        arrx[:,i] = np.array(norm(np.ndarray.tolist(arrx[:,i])))
    normx = np.ndarray.tolist(arrx)
    normy = norm(y)
    return (data,x,y,normx,normy)

def classify(x,y,normx,normy,n_comp,cov,norm):
    if norm:
        x = normx
        y = normy

    #X = np.array(x)
    pca = PCA(n_components=n_comp)
    x = pca.fit(x).transform(x)
    print(pca.explained_variance_ratio_)
    fitter = GaussianMixture(n_components=2,covariance_type=cov,n_init=10,max_iter=200).fit(x)
    print(fitter.lower_bound_)
    #labels = fitter.labels_
    #ax.scatter(X[:,0],X[:,1],X[:,2],c=labels.astype(np.float),edgecolor='k')
    return min(sum(abs(fitter.predict(x)-y)),sum(abs((1 - fitter.predict(x)) - y)))/len(x)

runEverything()














