import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from warnings import catch_warnings

from ase.io import read
from ase.build import molecule
from ase import Atoms
from dscribe.descriptors import SOAP, ACSF

import qml
from qml.fchl import generate_representation, get_local_kernels, get_atomic_kernels, get_atomic_symmetric_kernels
from qml.math import cho_solve
import scipy.sparse



def sparse_to_file(fpath, spA):
    # save sparse matrix to file
    file = open(fpath,'w')

    for i in range(spA.shape[0]):
        for j in spA[i].nonzero()[1]:
            file.write(str(i)+'\t'+str(j)+'\t'+str(spA[i,j])+'\n')
    file.close()

def extract_atoms(folderdir, filedir):
    #folder = "data/qm9/"
    #filedir= folder + "dsgdb9nsd_000001.xyz"
    # loop this guy:
    fpath = folderdir+filedir
    with open(fpath,'r') as f:
        strlist = f.readlines()
        n_atom = int(strlist[0])
        atoms = strlist[2:2+n_atom]
        symbols = []
        coords = np.zeros((n_atom, 3))
        atomdata = {"filename":filedir, "n_atom":n_atom, "symbols": [], "coords": np.zeros((n_atom, 3))}
        for i, atom in enumerate(atoms):
            atomstr = atom.split("\t")
            atomtype = atomstr[0]
            coord = np.array([float(c) for c in atomstr[1:4]])
            atomdata["coords"][i] = coord
            atomdata["symbols"].append(atomtype)
    return atomdata

def extract_SOAP():
    mypath = "/users/baribowo/Dataset/gdb9-14b/geometry/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # extract coords here:
    start = time.time() # timer
    mols = []
    errfiles = []
    for f in sorted(onlyfiles):
        try:
            mols.append(extract_atoms(mypath, f))
        except:
            errfiles.append(f)

    with open("data/qm9_error_soap.txt", "w") as f:
        for errf in errfiles:
            f.write(errf+"\n")


    structures = []
    for mol in mols:
        structures.append(Atoms(symbols=mol["symbols"], positions = mol["coords"]))

    print(len(structures))
    species = set(["H", "C", "N", "O", "F"])
    #for structure in structures:
    #    species.update(structure.get_chemical_symbols())

    print(species)

    soap = SOAP(
        species=species,
        periodic=False,
        rcut=6.,
        nmax=3,
        lmax=3,
        sigma=0.1,
        average="off", #"inner",
        sparse=False
    )

    # batch here:
    ndata= len(onlyfiles)
    bsize = 1000
    blength = ndata // bsize

    batches = []
    c = range(0, blength)
    for i in c:
        n = i*bsize
        batches.append([n, n+bsize])
    bend = batches[-1][-1]
    bendsize = ndata - (blength*bsize)
    batches.append([bend, bend+bendsize+2])
    print(batches)

    outfolder = "/users/baribowo/Dataset/gdb9-14b/soap/"
    if not exists(outfolder):
        makedirs(outfolder)

    for j, batch in enumerate(batches):
        print("batch number ",j)
        feature_vectors = soap.create(structures[batch[0]:batch[1]], n_jobs=4) # batch
        feature_vectors = np.array(feature_vectors)

        # save numpy array to files, each mol = 1 file:
        for i, mol in enumerate(mols[batch[0]:batch[1]]): # batch
            #np.savetxt(outfolder+mol["filename"]+'.txt', feature_vectors[i], delimiter='\t')
            sp = scipy.sparse.csc_matrix(feature_vectors[i])
            sparse_to_file(outfolder+mol["filename"], sp)
            print(mol["filename"], "done!", feature_vectors[i].shape)

    print("elapsed time = ", time.time()-start, "s")

def test_ACSF():
    # Setting up the ACSF descriptor
    acsf = ACSF(
        6.0,
        species=["H", "O"],
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g3_params = [1],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    )

    # Creating an atomic system as an ase.Atoms-object
    water = molecule("H2O")

    # Create MBTR output for the hydrogen atom at index 1
    acsf_water = acsf.create(water)

    print(acsf_water)
    print(acsf_water.shape)

def extract_ACSF():
    mypath = "/users/baribowo/Dataset/gdb9-14b/geometry/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # extract coords here:
    start = time.time() # timer
    mols = []
    errfiles = []
    for f in sorted(onlyfiles):
        try:
            mols.append(extract_atoms(mypath, f))
        except:
            errfiles.append(f)

    with open("data/qm9_error_acsf.txt", "w") as f:
        for errf in errfiles:
            f.write(errf+"\n")


    structures = []
    for mol in mols:
        structures.append(Atoms(symbols=mol["symbols"], positions = mol["coords"]))

    print(len(structures))
    species = set(["H", "C", "N", "O", "F"])
    #for structure in structures:
    #    species.update(structure.get_chemical_symbols())

    print(species)

    soap = SOAP(
        species=species,
        periodic=False,
        rcut=6.,
        nmax=3,
        lmax=3,
        sigma=0.1,
        average="off", #"inner",
        sparse=False
    )

    # batch here:
    ndata= len(onlyfiles)
    bsize = 1000
    blength = ndata // bsize

    batches = []
    c = range(0, blength)
    for i in c:
        n = i*bsize
        batches.append([n, n+bsize])
    bend = batches[-1][-1]
    bendsize = ndata - (blength*bsize)
    batches.append([bend, bend+bendsize+2])
    print(batches)

    outfolder = "/users/baribowo/Dataset/gdb9-14b/soap/"
    if not exists(outfolder):
        makedirs(outfolder)

    for j, batch in enumerate(batches):
        print("batch number ",j)
        feature_vectors = soap.create(structures[batch[0]:batch[1]], n_jobs=4) # batch
        feature_vectors = np.array(feature_vectors)

        # save numpy array to files, each mol = 1 file:
        for i, mol in enumerate(mols[batch[0]:batch[1]]): # batch
            #np.savetxt(outfolder+mol["filename"]+'.txt', feature_vectors[i], delimiter='\t')
            sp = scipy.sparse.csc_matrix(feature_vectors[i])
            sparse_to_file(outfolder+mol["filename"], sp)
            print(mol["filename"], "done!", feature_vectors[i].shape)

    print("elapsed time = ", time.time()-start, "s")



def extract_FCHL():
    # file op:
    fpath = "data/qm9_error.txt"
    with open(fpath,'r') as f: # errorlist
        strlist = f.read()
        strlist = strlist.split("\n")
        errfiles = strlist[:-1]

    mypath = "data/qm9/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and (f not in errfiles)] # remove errorfiles
    onlyfiles = sorted(onlyfiles) #[1:] this is for nonubuntu
    print(len(onlyfiles), onlyfiles[0])

    # extract coords here:
    start = time.time() # timer
    # make FCHL folder
    feature_folder = "data/FCHL"
    if not exists(feature_folder):
        makedirs(feature_folder)
    
    n_atom_QM9 = 29
    for f in sorted(onlyfiles):
        # extract features:
        mol = qml.Compound(xyz=mypath+f)#, qml.Compound(xyz="data/qm9/dsgdb9nsd_000002.xyz")
        mol.generate_fchl_representation(max_size=n_atom_QM9, cut_distance=8.0, neighbors=n_atom_QM9) # neighbours is only used if it has periodic boundary
        print(mol.name)
        

        # save each to nested folder, each file contains 5 x 29 matrix:
        mol_folder = feature_folder+"/"+f # generate molecule folder
        if not exists(mol_folder):
            makedirs(mol_folder)
        for i in range(n_atom_QM9):
            atom_folder = mol_folder+"/"+str(i) # gen atom folder
            np.savetxt(atom_folder+'.txt', mol.representation[i], delimiter='\t')
    print("elapsed time = ", time.time()-start, "s")


def train_FCHL():
    fpath = "data/qm9_error.txt"
    with open(fpath,'r') as f: # errorlist
        strlist = f.read()
        strlist = strlist.split("\n")
        errfiles = strlist[:-1]

    mypath = "data/qm9/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and (f not in errfiles)] # remove errorfiles
    onlyfiles = sorted(onlyfiles)[1:]
    onlyfiles = np.array(onlyfiles)
    print(onlyfiles)

    # load energies:
    E = np.loadtxt("data/energies.txt")
    Nqm9 = len(E)
    print(Nqm9)

    # centers:
    centers = np.loadtxt("data/sel_centers.txt", dtype=int)
    # determine indices:
    idtrain = centers
    idtest = np.setdiff1d(list(range(Nqm9)), idtrain)
    # reduce energy:
    E_red = np.loadtxt("data/atomic_energies.txt")
    E[idtrain] = E[idtrain] - E_red[idtrain]

    # TRAINING REGIMENT
    # compute features :
    n_atom_QM9 = 29
    cutoff = 8.0
    Xtrain = []
    for f in onlyfiles[idtrain]:
        # extract features:
        mol = qml.Compound(xyz=mypath+f)#, qml.Compound(xyz="data/qm9/dsgdb9nsd_000002.xyz")
        mol.generate_fchl_representation(max_size=n_atom_QM9, cut_distance=cutoff, neighbors=n_atom_QM9) # neighbours is only used if it has periodic boundary
        #print(mol.name)
        Xtrain.append(mol.representation)

    # generate kernels:
    Xtrain = np.array(Xtrain)
    sigmas = [32.] #[32.]
    Ktrain = get_local_kernels(Xtrain, Xtrain, sigmas, cut_distance=cutoff)
    print(Ktrain.shape)
    # solve model:
    alpha = cho_solve(Ktrain[0], E[idtrain])
    Y = np.dot(Ktrain[0], alpha)
    # return energy magnitude:
    Y = Y + E_red[idtrain]
    E[idtrain] = E[idtrain] + E_red[idtrain]
    print("MAEtrain = ", np.mean(np.abs(Y - E[idtrain]))*627.5)

    # TESTING REGIMENT:
    # compute kernel in batch:
    ndata = len(idtest)
    bsize = 10000
    blength = ndata // bsize

    batches = []
    c = range(0, blength)
    for i in c:
        n = i*bsize
        batches.append([n, n+bsize])
    bend = batches[-1][-1]
    bendsize = ndata - (blength*bsize)
    batches.append([bend, bend+bendsize+2])

    start = time.time() # timer
    Ktest = np.array([])
    for i, batch in enumerate(batches):
        print("batch number ", i, "range", batch[0], batch[1])
        Xtest = []
        for f in onlyfiles[idtest[batch[0]:batch[1]]]:
            # extract features:
            mol = qml.Compound(xyz=mypath+f)#, qml.Compound(xyz="data/qm9/dsgdb9nsd_000002.xyz")
            mol.generate_fchl_representation(max_size=n_atom_QM9, cut_distance=cutoff, neighbors=n_atom_QM9)
            #print(mol.name)
            Xtest.append(mol.representation)
        Xtest = np.array(Xtest)
        if Ktest.size == 0: # init kernel
            Ktest = get_local_kernels(Xtest, Xtrain, sigmas, cut_distance=cutoff)[0] # slice the first dim
        else: # stack kernel
            Kbatch = get_local_kernels(Xtest, Xtrain, sigmas, cut_distance=cutoff)[0]
            Ktest = np.vstack((Ktest, Kbatch))

    Y = np.dot(Ktest, alpha)
    Y = Y + E_red[idtest] # return energy magnitude
    print("batchpred t = ", time.time()-start, "s")
    print("MAEtest = ", np.mean(np.abs(Y - E[idtest]))*627.5)

def getatom_FCHL():
    fpath = "data/qm9_error.txt"
    with open(fpath,'r') as f: # errorlist
        strlist = f.read()
        strlist = strlist.split("\n")
        errfiles = strlist[:-1]

    mypath = "data/qm9/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and (f not in errfiles)] # remove errorfiles
    onlyfiles = sorted(onlyfiles)
    onlyfiles = np.array(onlyfiles)

    n_atom_QM9 = 29
    Xtrain = []
    for f in sorted(onlyfiles)[-2:-1]:
        # extract features:
        mol = qml.Compound(xyz=mypath+f)#, qml.Compound(xyz="data/qm9/dsgdb9nsd_000002.xyz")
        mol.generate_fchl_representation(max_size=n_atom_QM9, cut_distance=8.0, neighbors=n_atom_QM9) # neighbours is only used if it has periodic boundary
        Xtrain.append(mol.representation)
        print(mol.name)
    
    Xtrain = np.array(Xtrain)
    print(Xtrain[0, 0, :5, :5])
    cutoff = 8.
    sigmas = [32.]
    Ktrain = get_local_kernels(Xtrain, Xtrain, sigmas, cut_distance=cutoff)


# main:
test_ACSF()
