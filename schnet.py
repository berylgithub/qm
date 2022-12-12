from schnetpack.datasets import QM9

qm9data = QM9('./qm9.db', load_only=[QM9.U0], remove_uncharacterized=True)

atomrefs = qm9data.get_atomref(QM9.U0)
print('U0 of hyrogen:', '{:.2f}'.format(atomrefs[QM9.U0][1][0]), 'eV')
print('U0 of carbon:', '{:.2f}'.format(atomrefs[QM9.U0][6][0]), 'eV')
print('U0 of oxygen:', '{:.2f}'.format(atomrefs[QM9.U0][8][0]), 'eV')
