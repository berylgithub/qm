from ase.io import read
from ase.build import molecule
from ase import Atoms

from dscribe.descriptors import SOAP

# Let's use ASE to create atomic structures as ase.Atoms objects.
#structure1 = read("water.xyz")
structure2 = molecule("H2O")
structure3 = Atoms(symbols=["H", "C", "O", "N"], positions=[[0., 0., 0.], [1.128, 0., 0.], [2.5, 1.128, 0.], [-3., -1.5, 0.2]])

structures = [structure2, structure3]
species = set()
for structure in structures:
    species.update(structure.get_chemical_symbols())

print(species)

soap = SOAP(
    species=species,
    periodic=False,
    rcut=5,
    nmax=8,
    lmax=4,
    average="off",
    sparse=False
)

feature_vectors = soap.create(structures, n_jobs=1)
print(len(feature_vectors), feature_vectors[1].shape)

# save numpy array to files, each mol = 1 file: