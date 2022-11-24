from warnings import catch_warnings
from ase.io import read
from ase.build import molecule
from ase import Atoms
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from dscribe.descriptors import SOAP



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

species = set()
for structure in structures:
    species.update(structure.get_chemical_symbols())

print(species)

soap = SOAP(
    species=species,
    periodic=False,
    rcut=5,
    nmax=6,
    lmax=6,
    average="off",
    sparse=False
)

feature_vectors = soap.create(structures, n_jobs=4)
print(len(feature_vectors), feature_vectors[0].shape)
feature_vectors = np.array(feature_vectors)

# save numpy array to files, each mol = 1 file:
outfolder = "data/SOAP/"
if not exists(outfolder):
    makedirs(outfolder)

for i, mol in enumerate(mols):
    np.savetxt(outfolder+mol["filename"]+'.txt', feature_vectors[i], delimiter='\t')

print("elapsed time = ", time.time()-start, "s")