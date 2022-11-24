from ase.io import read
from ase.build import molecule
from ase import Atoms
import numpy as np

from dscribe.descriptors import SOAP

def extract_atoms():
    folder = "data/qm9/"
    filedir= folder + "dsgdb9nsd_000001.xyz"
    # loop this guy:
    with open(filedir,'r') as f:
        strlist = f.readlines()
        n_atom = int(strlist[0])
        print(strlist[2:2+n_atom])
        atoms = strlist[2:2+n_atom]
        symbols = []
        coords = np.zeros((n_atom, 3))
        atomdata = {"n_atom":n_atom, "symbols": [], "coords": np.zeros((n_atom, 3))}
        for i, atom in enumerate(atoms):
            atomstr = atom.split("\t")
            atomtype = atomstr[0]
            coord = np.array([float(c) for c in atomstr[1:4]])
            atomdata["coords"][i] = coord
            atomdata["symbols"].append(atomtype)
        print(atomdata)
    return None

# Let's use ASE to create atomic structures as ase.Atoms objects.
#structure1 = read("data/qm9/dsgdb9nsd_000001.xyz")
structure2 = molecule("H2O")
structure3 = Atoms(symbols=["H", "C", "O", "N"], positions = np.array([[0., 0., 0.], [1.128, 0., 0.], [2.5, 1.128, 0.], [-3., -1.5, 0.2]]))

structures = [structure2, structure3]
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

feature_vectors = soap.create(structures, n_jobs=1)
print(len(feature_vectors), feature_vectors[0].shape)
feature_vectors = np.array(feature_vectors)
print(feature_vectors[0].shape)
# save numpy array to files, each mol = 1 file:


extract_atoms()