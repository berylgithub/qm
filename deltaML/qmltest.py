import qml
from qml.kernels import gaussian_kernel
from qml.math import cho_solve

import numpy as np
import random
import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from warnings import catch_warnings

# Follow the QML tutorial until delta learning:

# data setup
geopath = "/users/baribowo/Dataset/tutorial/qm7"
compounds = [qml.Compound(xyz=geopath+"/"+f) for f in sorted(os.listdir(geopath))]

Ehofs = []; Edftbs = []; Edeltas = []
efile = "/users/baribowo/Dataset/tutorial/hof_qm7.txt"
f = open(efile, "r")
lines = f.readlines()
f.close()
for line in lines:
    tokens = line.split()
    molname = tokens[0]
    Ehof = float(tokens[1])
    Edftb = float(tokens[2])
    Edelta = Ehof - Edftb
    Ehofs.append(Ehof); Edftbs.append(Edftb); Edeltas.append(Edelta)
Ehofs = np.array(Ehofs); Edftbs = np.array(Edftbs); Edeltas = np.array(Edeltas)
# extract features while also saving it to text file for Julia later
for mol in compounds:
    mol.generate_coulomb_matrix(size=23, sorting="row-norm")

X = np.array([mol.representation for mol in compounds])
np.savetxt("/users/baribowo/Dataset/qm7coulomb.txt", X, delimiter="\t") #write to file for Julia purposes
random.seed(603)
Ndata = len(compounds)
idtrain = random.sample(range(Ndata), 1000)
idtest = np.setdiff1d(list(range(Ndata)), idtrain)
print(Ndata, len(idtrain), len(idtest))

Xtrain = X[idtrain]; Xtest = X[idtest]
sigma = 700.

# fit and test standard QM7 curve
Ytrain = Ehofs[idtrain]; Ytest = Ehofs[idtest] 
K = gaussian_kernel(Xtrain, Xtrain, sigma)
K[np.diag_indices_from(K)] += 1e-8

alpha = cho_solve(K, Ytrain)
K = gaussian_kernel(Xtest, Xtrain, sigma)
Ypred = K@alpha
print(np.mean(np.abs(Ypred - Ytest)))

# fit and test delta curve
# see if E = deltaE + Ebase is more accurate