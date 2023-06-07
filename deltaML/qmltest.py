import qml
import numpy as np
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
onlyfiles = [f for f in listdir(geopath) if isfile(join(geopath, f))] # remove errorfiles
onlyfiles = sorted(onlyfiles)
print(onlyfiles)
# fit and test standard QM7 curve
# fit and test delta curve
# see if E = deltaE + Ebase is more accurate