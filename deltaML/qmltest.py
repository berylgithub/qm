import qml
import numpy as np
import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from warnings import catch_warnings

fpath = "/users/baribowo/Dataset/tutorial/qm7/0001.xyz"

mol = qml.Compound(xyz = fpath)

mol.generate_coulomb_matrix(size=5, sorting="row-norm")
print(mol.representation)

# Follow the QML tutorial until delta learning:
# extract features while also saving it to text file for Julia later
geopath = "/users/baribowo/Dataset/tutorial/qm7"
compounds = [qml.Compound(xyz=geopath+f) for f in sorted(os.listdir(geopath))]
print(compounds[1])
# fit and test standard QM7 curve


# fit and test delta curve
# see if E = deltaE + Ebase is more accurate