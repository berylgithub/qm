import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from warnings import catch_warnings

""" from ase.io import read
from ase.build import molecule
from ase import Atoms
from dscribe.descriptors import SOAP """

import qml
from qml.fchl import generate_representation


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
    mypath = "data/qm9/"
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
        print(mol["filename"])
        structures.append(Atoms(symbols=mol["symbols"], positions = mol["coords"]))

    species = set(["H", "C", "N", "O", "F"])
    #for structure in structures:
    #    species.update(structure.get_chemical_symbols())

    print(species)

    soap = SOAP(
        species=species,
        periodic=False,
        rcut=5,
        nmax=5,
        lmax=5,
        average="inner",
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

    outfolder = "data/SOAP/"
    if not exists(outfolder):
        makedirs(outfolder)

    for i, batch in enumerate(batches):
        print("batch number ",i)
        feature_vectors = soap.create(structures[batch[0]:batch[1]], n_jobs=4) # batch
        feature_vectors = np.array(feature_vectors)

        # save numpy array to files, each mol = 1 file:
        for i, mol in enumerate(mols[batch[0]:batch[1]]): # batch
            np.savetxt(outfolder+mol["filename"]+'.txt', feature_vectors[i], delimiter='\t')

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

# main:
extract_FCHL()