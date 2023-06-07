import qml
import numpy as np
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
lines = f.readlines
f.close()
for line in lines:
    tokens = line.split()
    molname = tokens[0]
    Ehof = float(tokens[1])
    Edftb = float(tokens[2])
    Edelta = Ehof - Edftb
    Ehofs.append(Ehof); Edftbs.append(Edftb); Edeltas.append(Edelta)

print(Ehofs)
print(Edftbs)
print(Edeltas)
# extract features while also saving it to text file for Julia later


# fit and test standard QM7 curve


# fit and test delta curve
# see if E = deltaE + Ebase is more accurate