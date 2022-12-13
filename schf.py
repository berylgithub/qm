import os
import schnetpack as spk
from schnetpack.datasets import QM9

qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

qm9data = QM9('./qm9.db', load_only=[QM9.U0], remove_uncharacterized=True) # already pre-downloaded
#print(qm9data.get_properties(0))
#print(len(qm9data))

train, val, test = spk.train_test_split(
        data=qm9data,
        num_train=100,
        num_val=10,
        split_file=os.path.join(qm9tut, "split.npz"),
    )

#print(qm9data[0]) # probably this is how to feed the data to model THIS IS A DICTIONARY!!
fulldata = spk.AtomsLoader([qm9data[0]], batch_size=1, shuffle=False)

atomrefs = qm9data.get_atomref(QM9.U0) # tensor of atom energy
means, stddevs = fulldata.get_statistics(
    QM9.U0, divide_by_atoms=True, single_atom_ref=atomrefs
)



sch_feat = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff, return_intermediate = True
)


for batch in fulldata:
    f = sch_feat.forward(batch)
    #print(batch)
    print(f)