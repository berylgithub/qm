import qml

fpath = "/users/baribowo/Dataset/tutorial/qm7/0001.xyz"

mol = qml.Compound(xyz = fpath)

mol.generate_coulomb_matrix(size=5, sorting="row-norm")
print(mol.representation)