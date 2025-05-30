import qml
from qml.kernels import gaussian_kernel
from qml.representations import get_slatm_mbtypes
from qml.math import cho_solve
import numpy as np
import scipy.sparse

import random
import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from warnings import catch_warnings

# Follow the QML tutorial until delta learning:

# data setup

def test_qml_deltaML():
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
    sigma = 700.

    # fitting with incremental dataset for training
    Ntrain = [1000, 2000, 4000]
    output = np.zeros((len(Ntrain), 3)) # (ntrain, MAE target, MAE delta)
    for i,n in enumerate(Ntrain):
        # data indexing
        idtrain = random.sample(range(Ndata), n)
        idtest = np.setdiff1d(list(range(Ndata)), idtrain)
        print("num of (total, train, test) data = ",Ndata, len(idtrain), len(idtest))
        Xtrain = X[idtrain]; Xtest = X[idtest]

        # fit and test standard QM7 curve
        Ytrain = Ehofs[idtrain]; Ytest = Ehofs[idtest] 
        K = gaussian_kernel(Xtrain, Xtrain, sigma)
        K[np.diag_indices_from(K)] += 1e-8
        alpha = cho_solve(K, Ytrain)
        K = gaussian_kernel(Xtest, Xtrain, sigma)
        Ypred = K@alpha
        MAEtot = np.mean(np.abs(Ypred - Ytest))
        print("MAE Etot = ", MAEtot)

        # fit and test delta curve
        Ytrain = Edeltas[idtrain]; Ytest = Edeltas[idtest] 
        K = gaussian_kernel(Xtrain, Xtrain, sigma)
        K[np.diag_indices_from(K)] += 1e-8
        alpha = cho_solve(K, Ytrain)
        K = gaussian_kernel(Xtest, Xtrain, sigma)
        Ypred = K@alpha
        MAEdelta = np.mean(np.abs(Ypred - Ytest))
        print("MAE Edelta = ", MAEdelta)

        np.savetxt("train_indexes_"+str(n)+".txt", idtrain, delimiter="\n")
        output[i,0] = n; output[i,1] = MAEtot; output[i,2] = MAEdelta

        # see if E = deltaE + Ebase is more accurate
        #Etot = Ehofs[idtest]; Ebase = Edftbs[idtest]; Edelta = Ypred; 
        #Etotpred = Ebase + Edelta
        #print("MAE Etarget = ", np.mean(np.abs(Etotpred - Etot)))
    np.savetxt("qm7_MAE.txt", output, delimiter="\t")

def sparse_to_file(fpath, spA):
    # save sparse matrix to file
    file = open(fpath,'w')

    for i in range(spA.shape[0]):
        for j in spA[i].nonzero()[1]:
            file.write(str(i)+'\t'+str(j)+'\t'+str(spA[i,j])+'\n')
    file.close()

def extract_QML_features():
    # extract features then save to file, takes in path of the geometries and outputs the features in text file
    #geopath = "/users/baribowo/Dataset/zaspel_supp/supplementary/geometry" # OMP1 geometry filepath
    geopath = "/users/baribowo/Dataset/gdb9-14b/geometry/"
    onlyfiles = sorted([f for f in listdir(geopath) if isfile(join(geopath, f))])
    print("Ndata = ",len(onlyfiles))
    compounds = [qml.Compound(xyz=geopath+f) for f in onlyfiles]
    #mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
    ncs = [(mol.nuclear_charges) for mol in compounds]
    elements = np.unique(np.concatenate(ncs))
    print(elements)
    """ coor = compounds[0].coordinates
    nc = compounds[0].nuclear_charges
    rep = qml.representations.generate_fchl_acsf(nc, coor, gradients=False, elements=elements, nRs2=12, nRs3=10, rcut=6)
    print(rep.shape)
    print(rep) 
    sp = sparse_matrix = scipy.sparse.csc_matrix(rep) 
    # np.savetxt("/users/baribowo/Dataset/gdb9-14b/fchl19/0.txt", rep, delimiter="\t")
    # sparse_to_file('/users/baribowo/Dataset/gdb9-14b/fchl19/0_sparse.txt', sp) """
    for i, mol in enumerate(compounds):
        molid = i+1
        #if molid == 184: # not sure why mol num 184 cause error for acsf, check later
        #    continue
        coor = mol.coordinates
        nc = mol.nuclear_charges
        #rep = qml.representations.generate_fchl_acsf(nc, coor, gradients=False, elements=elements, nRs2=12, nRs3=10, rcut=6)
        rep = qml.representations.generate_acsf(nc, coor, gradients=False, rcut=6)
        sp = scipy.sparse.csc_matrix(rep)
        print(molid, rep.shape)
        sparse_to_file('/users/baribowo/Dataset/gdb9-14b/acsf/'+onlyfiles[i], sp)
        #mol.generate_slatm(mbtypes, local=True)
        #mol.generate_coulomb_matrix(size=23, sorting="row-norm")
    #X = np.array([mol.representation for mol in compounds])
    #print(compounds[0].representation.shape)
    #np.savetxt("/users/baribowo/Dataset/coulomb_zaspel.txt", X, delimiter="\t")



#test_qml_deltaML()
extract_QML_features()