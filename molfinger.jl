using ASE, ACSF

function test_ASE()
    at = bulk("Si")
    #at1 = ASEAtoms("N2", positions)
    out = acsf(at)
    display(out)
    display(size(out[1]))
end

test_ASE()