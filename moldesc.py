import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from warnings import catch_warnings

#from ase.io import read
#from ase.build import molecule
#from ase import Atoms
#from dscribe.descriptors import SOAP, ACSF

import scipy.sparse
import qml
#from qml.fchl import generate_representation, get_local_kernels, get_atomic_kernels, get_atomic_symmetric_kernels
#from qml.math import cho_solve
import MBDF
#from cMBDF_joblib import get_cmbdf
import cMBDF_joblib


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
    species = ["H", "C", "N", "O", "F"]
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
    g4eta = [   1.0,    1.0]
    g4lambda = [  -1.0,    1.0]
    g4zeta =  [   1.0,    1.0]
    g4 = np.zeros((len(g4eta), 3))
    for i, _ in enumerate(g4eta):
        g4[i, 0] = g4eta[i]
        g4[i, 1] = g4zeta[i]
        g4[i, 2] = g4lambda[i]
    
    acsf = ACSF(
        6.0,
        species=["H", "O"],
        g2_params=[[9.0, 1.],  [100.0, 1.]],
        g3_params=[1, 2],
        g4_params=g4,
        g5_params=g4,
    )
    # Creating an atomic system as an ase.Atoms-object
    water = molecule("H2O")
    # Create MBTR output for the hydrogen atom at index 1
    acsf_water = acsf.create(water)
    print(acsf_water)
    print(acsf_water.shape)

    acsf = ACSF(
        6.0,
        species=["O", "H"],
        g2_params=[[9.0, 1.],  [100.0, 1.]],
        g3_params=[1, 2],
        g4_params=g4,
        g5_params=g4,
    )
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

    # g4s from descriptorzoo
    g4eta = [   1.0,    1.0,    1.0,    1.0,  
           80.0,   80.0,   80.0,   80.0,    250.0,  250.0,  250.0,  250.0, 
              800.0,  800.0,  800.0,  800.0, 
          ]
    g4lambda = [  -1.0,    1.0,   -1.0,    1.0,  
           -1.0,    1.0,   -1.0,    1.0,      -1.0,    1.0,   -1.0,    1.0,  
                 -1.0,    1.0,   -1.0,    1.0,
           ]
    g4zeta =  [   1.0,    1.0,    2.0,    2.0, 
            1.0,    1.0,    2.0,    2.0,        1.0,    1.0,    2.0,    2.0,   
                    1.0,    1.0,    2.0,    2.0,
            ]
    g4 = np.zeros((len(g4eta), 3))
    for i, _ in enumerate(g4eta):
        g4[i, 0] = g4eta[i]
        g4[i, 1] = g4zeta[i]
        g4[i, 2] = g4lambda[i]
    
    acsf = ACSF(
        6.0,
        species=species,
        g2_params=[[9.0, 1.],  [100.0, 1.], [1000.0, 1.], [4000.0, 1.]],
        g3_params=[1, 2],
        g4_params=g4,
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

    outfolder = "/users/baribowo/Dataset/gdb9-14b/acsf/"
    if not exists(outfolder):
        makedirs(outfolder)

    for j, batch in enumerate(batches):
        print("batch number ",j)
        feature_vectors = acsf.create(structures[batch[0]:batch[1]], n_jobs=4) # batch
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

def test_MBDF():
    geopath = "/users/baribowo/Dataset/gdb9-14b/geometry/"
    ntest = 5
    onlyfiles = sorted([f for f in listdir(geopath) if isfile(join(geopath, f))])[:ntest]
    print("Ndata = ",len(onlyfiles))
    compounds = [qml.Compound(xyz=geopath+f) for f in onlyfiles]
    coors = np.array([mol.coordinates for mol in compounds])
    #mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
    ncs = np.array([mol.nuclear_charges for mol in compounds])
    elements = np.unique(np.concatenate(ncs))
    #print(ncs)
    #print(coors)
    #mbdf = cMBDF.generate_mbdf(ncs, coors)
    start = time.time() # timer
    reps = cMBDF_joblib.generate_mbdf(ncs, coors, gradients=False, progress_bar = False, n_atm=2.0)
    print(time.time()-start)
    print(reps)
    print(reps.shape)
    # write to file:
    #for i, elem in enumerate(reps):
    #    np.savetxt("/users/baribowo/Dataset/gdb9-14b/cmbdf-2/"+str(i+1)+".txt", elem, delimiter="\t")



def extract_MBDF():
    geopath = "/users/baribowo/Dataset/gdb9-14b/geometry/"
    onlyfiles = sorted([f for f in listdir(geopath) if isfile(join(geopath, f))])
    print("Ndata = ",len(onlyfiles))
    compounds = [qml.Compound(xyz=geopath+f) for f in onlyfiles]
    coors = np.array([mol.coordinates for mol in compounds])
    #mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
    ncs = np.array([mol.nuclear_charges for mol in compounds])
    elements = np.unique(np.concatenate(ncs))
    #print(ncs)
    #print(coors)
    #mbdf = cMBDF.generate_mbdf(ncs, coors)
    start = time.time() # timer
    reps = cMBDF_joblib.generate_mbdf(ncs, coors, gradients=False, progress_bar = False, n_atm=2.0)
    print(time.time()-start)
    #print(mbdf)
    #print(mbdf.shape)
    # write to file:
    for i, elem in enumerate(reps):
        np.savetxt("/users/baribowo/Dataset/gdb9-14b/cmbdf-2/"+str(i+1)+".txt", elem, delimiter="\t")

# extracts CM and/or BOB using MBDF.py script
def extract_CM():
    geopath = "/home/berylubuntu/Dataset/gdb9-14b/geometry/"
    onlyfiles = sorted([f for f in listdir(geopath) if isfile(join(geopath, f))])
    print(onlyfiles)
    print("Ndata = ",len(onlyfiles))
    compounds = [qml.Compound(xyz=geopath+f) for f in onlyfiles]
    coors = [mol.coordinates for mol in compounds]
    #mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
    ncs = [mol.nuclear_charges for mol in compounds]
    elements = np.unique(np.concatenate(ncs))
    #print(ncs)
    #print(coors)
    #print(elements)
    #mbdf = cMBDF.generate_mbdf(ncs, coors)
    start = time.time() # timer
    #reps = cMBDF_joblib.generate_mbdf(ncs, coors, gradients=False, progress_bar = False, n_atm=2.0)
    #reps = MBDF.generate_CM(coors,ncs,10)
    _ = [mol.generate_coulomb_matrix(size=29, sorting="row-norm") for mol in compounds]
    #_ = [mol.generate_bob(asize={"C":5, "H":5, "O":3}) for mol in compounds]
    #print([mol.representation for mol in compounds])
    print(time.time()-start)
    for i, mol in enumerate(compounds):
        np.savetxt("/home/berylubuntu/Dataset/gdb9-14b/cm/"+str(i+1)+".txt", mol.representation, delimiter="\t")
        print(i, "done!")

def extract_BOB():
    geopath = "/home/berylubuntu/Dataset/gdb9-14b/geometry/"
    onlyfiles = sorted([f for f in listdir(geopath) if isfile(join(geopath, f))])[:5]
    print(onlyfiles)
    print("Ndata = ",len(onlyfiles))
    compounds = [qml.Compound(xyz=geopath+f) for f in onlyfiles]
    coors = [mol.coordinates for mol in compounds]
    #mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
    ncs = [mol.nuclear_charges for mol in compounds]
    elements = np.unique(np.concatenate(ncs))
    atoms = nc_to_atype(ncs) 
    #print(ncs)
    #print(coors)
    print(elements)
    #mbdf = cMBDF.generate_mbdf(ncs, coors)
    start = time.time() # timer
    #reps = cMBDF_joblib.generate_mbdf(ncs, coors, gradients=False, progress_bar = False, n_atm=2.0)
    #reps = MBDF.generate_CM(coors,ncs,10)
    #_ = [mol.generate_coulomb_matrix(size=29, sorting="row-norm") for mol in compounds]
    _ = [mol.generate_bob(asize={"C":9, "H":20, "N":7, "O":5, "F":6}) for mol in compounds]
    #reps = MBDF.generate_bob(atoms, coors, asize={"C":9, "H":20, "N":7, "O":5, "F":6})
    #print([mol.representation for mol in compounds])
    print(time.time()-start)
    for mol in compounds:
        print(mol.representation.shape)

def test_CM_BOB():
    ncs = [[6,1,8],[6,1,7]]
    atoms = nc_to_atype(ncs)
    print(atoms)
    #atoms = [["C", "H"],["C", "H"]]
    coors = [
                [[1.,1.,0.2],[0.1,0.1,0.1],[0.2,0.3,0.5]],
                [[1.,1.,0.2],[0.1,0.1,0.1], [2.,3.,5.]]
            ]
    #reps = MBDF.generate_CM(coors,ncs,5)
    #reps = MBDF.generate_bob(atoms, coors, asize={"C":9, "H":20, "N":7, "O":5, "F":6})
    reps = qml.representations.generate_bob([[6,1,8]], [[1.,1.,0.2],[0.1,0.1,0.1],[0.2,0.3,0.5]], ["C","H","O"], asize={"C":1, "H":1, "N":1, "O":1})
    for rep in reps:
        print(rep.shape)
        print(rep)

def nc_to_atype(ncs):
    atoms = []
    ncd = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F"}
    for i in range(len(ncs)):
        temp = []
        for j in range(len(ncs[i])):
            temp.append(ncd[ncs[i][j]])
        atoms.append(temp)            
        temp = []
    return atoms

# main:
#extract_ACSF()
#extract_MBDF()
#test_MBDF()
#test_CM_BOB()
extract_BOB()
