
using DelimitedFiles, DataStructures, HDF5, JLD


"""
generate molecular formula based on the list of atoms
the formula is in the form of ordered atoms: H_C_N_O_F
"""
function generate_mol_formula(atoms)
    c = counter(atoms)
    out = ""
    if c["H"] > 0
        out *= "H"*string(c["H"])
    end
    if c["C"] > 0
        out *= "C"*string(c["C"])
    end
    if c["N"] > 0
        out *= "N"*string(c["N"])
    end
    if c["O"] > 0
        out *= "O"*string(c["O"])
    end
    if c["F"] > 0
        out *= "F"*string(c["F"])
    end
    return out
end

"""
usual main caller
"""
function main()
    # read gdb file:
    fd = readdlm("data/qm9/dsgdb9nsd_022908.xyz", '\t', String, '\n')
    n_atom = parse(Int64, fd[1,1])
    atoms = fd[3:3+n_atom-1, 1]
    energy = parse(Float64, fd[2, 13])
    coords = parse.(Float64, fd[3:3+n_atom-1, 2:4])
    formula = generate_mol_formula(atoms)
    data = Dict("n_atom" => n_atom, "formula" => formula, "atoms" => atoms, 
                "energy" => energy, "coordinates" => coords)
    

end

main()