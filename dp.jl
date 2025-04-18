"""
!!! TRANSFORM THE COORDS TO HARTREE!

The collection of functions for (d)ata (p)reparation
"""


using DelimitedFiles, DataStructures, JSON, JLD, BenchmarkTools, Printf
using Graphs, MolecularGraph, Combinatorics, SparseArrays # stuffs for ΔML
using LinearAlgebra
using ThreadsX
using PyCall

include("utils.jl") # bunch of utility functions


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
function data_transformer()
    # read gdb file:
    # using JLD || use HDF5 (later when performance is critical!!) || or even JuliaDB!!

    list_data = []
    list_error_file = []
    dir = "/users/baribowo/Dataset/gdb9-14b/geometry/"
    files = readdir(dir)
    count = 0
    # loop all data here:
    for f ∈ files
        if f[1] == '.' # skip anomalies
            continue
        end
        #= if count == 500 # limiter
            break
        end =#
        #println(f)
        id = f[11:16]
        println(id)
        fd = readdlm(dir*f,  '\t', String, '\n')
        n_atom = parse(Int64, fd[1,1])
        atoms = fd[3:3+n_atom-1, 1]
        energy = parse(Float64, fd[2, 13])
        coords = nothing
        try
            coords = parse.(Float64, fd[3:3+n_atom-1, 2:4])    
        catch e
            if e isa ArgumentError
                println("ArgumentError ", f)
                push!(list_error_file, f) 
                continue
            end
        end
        formula = generate_mol_formula(atoms)
        data = Dict("id" => id, "n_atom" => n_atom, "formula" => formula, "atoms" => atoms, 
                "energy" => energy, "coordinates" => coords)
        push!(list_data, data)
        count += 1
    end
    save("data/qm9_dataset.jld", "data", list_data)
    save("data/qm9_error_files.jld", "data", list_error_file)
    println("total data passed ", length(load("data/qm9_dataset.jld")["data"]))
    println("total data errors ", length(list_error_file))
    println("errors: ")
    println(load("data/qm9_error_files.jld")["data"])
end


"""
select data subset randomly, for faster prototyping
"""
function rand_select()
    # take 1000 data:
    num_subset = 1000
    dataset = load("data/qm9_dataset.jld")["data"]
    lendata = length(dataset)
    indices = rand(1:lendata, num_subset)
    selected_data = dataset[indices]
    save("data/qm9_dataset_$num_subset.jld", "data", selected_data)
    println(length(load("data/qm9_dataset_$num_subset.jld")["data"]))
end

"""
to accomodate unnecessary cell boundary in ASE package (library), by taking the maximum coord possible
"""
function get_max_cellbound()
    t = @elapsed begin
        dataset = load("data/qm9_dataset.jld")["data"]
    end
    println("elapsed ", t)
    max_coor = -Inf
    for d ∈ dataset
        max_c = maximum(d["coordinates"])
        if max_c > max_coor
            max_coor = max_c
        end
    end
    println(max_coor)
end


"""
transform SOAP filetexts to binary
"""
function getSOAPmol()
    path = "data/SOAP/"
    files = readdir(path)
    ndata = length(files)
    nf = size(readdlm(path*files[1]), 1) # length o0f the feature
    F = zeros(nf, ndata)
    start = time()
    @simd for i in eachindex(files)
        println(files[i]," done!!")
        @inbounds F[:, i] .= readdlm(path*files[i])
    end
    elapsed = time()-start
    display(F)
    display(Base.summarysize(F)*1e-6)
    println("elapsed time = ", elapsed)
    save("data/SOAP_mol.jld", "data", F') # transpose the matrix
end

function getFCHL()
    natom = 29; nf = 5
    path = "data/FCHL/"
    mols = readdir(path) #mols = ["dsgdb9nsd_000003.xyz", "dsgdb9nsd_000004.xyz"] # these are folders
    tail = ".txt"
    F = zeros(length(mols), natom, nf, natom)
    for l ∈ eachindex(mols)
        println(mols[l], " completed!!")
        for i ∈ (0:natom-1)
            str = path*mols[l]*"/"*string(i)*tail
            F[l, i+1, :, :] .= readdlm(str)
        end
    end
    save("data/FCHL.jld", "data", F)
end


function getSOAP()
    pth = "data/SOAP/"
    #str = "dsgdb9nsd_000001.xyz.txt"
    mols = readdir(pth)
    nf = 165; nmol = length(mols)
    f = Vector{Matrix{Float64}}(undef, nmol)
    for l ∈ eachindex(mols) 
        println(mols[l], " completed!!")
        f[l] = readdlm(pth*mols[l])
    end
    save("data/SOAP.jld", "data", f)
end

function getMBDF()
    folderpath = "/users/baribowo/Dataset/gdb9-14b/cmbdf-2/"
    dataset = load("data/qm9_dataset.jld", "data")
    files = readdir(folderpath) # the format is lexicographic here instead of standard numbering, i.e., 10 < 2
    nmol = length(files)
    ids = 1:nmol
    nf = 40 #6
    f = Vector{Matrix{Float64}}(undef, nmol)
    for i ∈ ids
        filepath = folderpath*string(i)*".txt"
        f[i] = readdlm(filepath)
        println(filepath, " done!!")
    end
    # slice rows by exids:
    exids = vec(Int.(readdlm("data/exids.txt")))
    f = f[setdiff(ids, exids)]
    display(f)
    # remove zeros using dataset:
    for i ∈ eachindex(f)
        natom = dataset[i]["n_atom"]
        f[i] = f[i][1:natom, :]
    end
    save("data/CMBDF2.jld", "data", f) # save to file
end

function table_results(foldername)
    sheader = ["n_data", "n_af", "n_mf", "n_basis", "num_centers", "ft_sos", "ft_bin"]
    rheader = ["Nqm9", "nK", "nU", "n_feature", "n_basis", "MAE", "RMSD", "max(MAD)", "t_ab", "t_ls", "t_batch"]
    outheader = ["nK", "nU", "n_af", "n_mf", "n_basis", "MAE", "ft_sos", "ft_bin", "t_solver", "t_pred"]
    setup_info = readdlm("data/$foldername/setup_info.txt", '\t', '\n')
    res_info = readdlm("result/$foldername/err_$foldername.txt", '\t', '\n')
    #display(setup_info)
    #display(res_info)
    out = Dict()
    out["nK"] = Integer.(res_info[:, 2])
    out["nU"] = Integer.(res_info[:, 3])
    out["n_af"] = setup_info[:, 2]
    out["n_mf"] = Integer.(res_info[:, 4])
    out["n_basis"] = Integer.(res_info[:, 5])
    out["MAE"] = res_info[:, 6]
    out["ft_sos"] = setup_info[:, 6]
    out["ft_bin"] = setup_info[:, 7]
    out["t_solver"] = round.(res_info[:, 11])
    out["t_pred"] = round.(res_info[:, 12])
    #display(out)
    open("result/$foldername/output_table.txt","w") do io
        str = ""
        for i ∈ eachindex(outheader)
            str *= outheader[i]*"\t"
        end
        print(io, str*"\n")
    end
    open("result/$foldername/output_table.txt","a") do io
        writedlm(io, [out["nK"] out["nU"] out["n_af"] out["n_mf"] out["n_basis"] out["MAE"] out["ft_sos"] out["ft_bin"] out["t_solver"] out["t_pred"]])
    end
end


function timeload()
    # took 2 minutes to load 130k data
    t = @elapsed begin
        load("data/qm9_dataset.jld")["data"][1]
    end
    println(t)
end

"""
========== ΔML stuffs ==========
"""

function getDistances(R) # copied from ΔML
    # computes interatomic distances given coordinates
    # R row := number of atoms
    natom = size(R, 1)
    D = zeros(natom, natom)
    ids = 1:natom
    for c ∈ Iterators.product(ids, ids)
        D[c[1],c[2]] = norm(R[c[1],:]-R[c[2],:])
    end
    return D
end


"""
function to prepare nuclear charges and distances, change logic whenever needed (e.g., if the dataset is different)
"""
function generate_charges_distances()
    # readpath and prep info
    geopath = "../../Dataset/zaspel_supp/geometry/" # geometry filepath, can be used for other dataset too as long as the format is the same
    ncpath = "deltaML/data/nuclear_charges.txt" # 
    geos = readdir(geopath)
    ncinfo = readdlm(ncpath)
    ncdict = Dict()
    for (i, nckey) ∈ enumerate(ncinfo[:, 1])
        ncdict[nckey] = ncinfo[:, 2][i]
    end
    # get Z and D and store in file
    ndata = length(geos)
    moldata = Vector{Any}(undef, ndata) # list of dicts
    Z = Dict(); R = zeros(ndata, 3)
    molinfos = map(geo -> readdlm(geopath*geo), geos)
    for i ∈ eachindex(molinfos)
        moldata[i] = Dict()
        moldata[i]["nc"] = map(j -> ncdict[molinfos[i][j, 1]], 2:molinfos[i][1,1]+1) # fill nuclear charges, probably will need to be changed depending on the geometry format
        moldata[i]["d"] = getDistances(molinfos[i][2:end,2:end])
    end
    save("deltaML/data/zaspel_ncd.jld", "data", moldata)
end


"""
for fchl, but can be used for general sparse
    nf: 
        fchl = 360, 
        soap = 480,
        acsf = 315
"""
function load_sparse()
    fpath = "/users/baribowo/Dataset/gdb9-14b/acsf/"
    files = readdir(fpath)
    ndata = length(files); nf = 315 # need to know the length of features (column) beforehand
    A = []
    for file ∈ files
        spg = readdlm(fpath*file)
        rows = spg[:, 1] .+ 1; cols = spg[:, 2] .+ 1; entries = spg[:, 3] # the +1s due to py indexing
        sp_temp = sparse(rows, cols, entries)
        natom = size(sp_temp, 1); ncols = size(sp_temp, 2)
        spA = spzeros(natom, nf) # s.t. the columns are equal accross all dataset
        spA[:, 1:ncols] = sp_temp
        push!(A, spA)
        println(file, "done!!")
    end
    save("data/ACSF.jld", "data", A)
end

"""
load features that are in vector form for each molecule -> transform into a big matrix
sizes:
    - CM = 435
"""
function load_vector_feature()
    homepath = "/home/berylubuntu/"
    fpath = homepath*"Dataset/gdb9-14b/cm/" # can also be "bob"
    files = readdir(fpath)
    A = zeros(length(files), 435)
    for (i,fil) in enumerate(files)
        A[i,:] = vec(readdlm(fpath*fil))
    end
    display(A)
    save("data/CM.jld", "data", A)
end


"""
remove the molids of uncharacterized ∪ non converged geos
"""
function feature_slicer()
    slicer = vec(Int.(readdlm("data/exids.txt")))
    #paths = ["data/qm9_dataset.jld", "data/FCHL19.jld", "data/SOAP.jld", "data/SOAP.jld"] 
    #feature_paths = ["data/atomref_features.jld", "data/featuresmat_qm9_covalentbonds.jld"] # all dataset then features
    feature_paths = ["data/CM.jld"]
    for i ∈ eachindex(feature_paths)
        println("proc ",feature_paths[i], " ...")
        t = @elapsed begin
            F = load(feature_paths[i], "data")
            ndata = size(F, 1) #nrow
            sliced = setdiff(1:ndata, slicer)
            save("OLD_"*feature_paths[i], "data", F) # unsliced
            if length(size(F)) > 1 # sliced:
                save(feature_paths[i], "data", F[sliced, :])
            else
                save(feature_paths[i], "data", F[sliced])
            end 
        end
        println("proc finished! ", t)
    end
end

"""
turn sparse features to dense
"""
function sparse_to_dense()
    fnames = ["ACSF", "SOAP", "FCHL19"]
    for fname ∈ fnames
        fpath = "data/"*fname*".jld"
        f = load(fpath, "data")
        ndata = length(f)
        idx = 1:ndata
        for i ∈ idx
            f[i] = Matrix(f[i])
        end
        outpath = "data/"*fname*"_dense.jld"
        save(outpath, "data", f)
    end
end

"""
SMILES util
"""
function fetch_SMILES(fpath)
    content = readdlm(fpath)
    natom = content[1,1]
    smiles = content[natom+4, 1]
    return smiles
end

"""
=== DRESSED BONDS ===
(prototype) get the bond order given smiles string of a molecule
returns dict of bondtype => count
"""


function get_bonds_from_SMILES(bondtypes, str; include_hydrogens=false)
    mol = smilestomol(str)
    if include_hydrogens
        add_hydrogens!(mol)
    end
    md = Dict()
    for key ∈ bondtypes # init dict
        md[key] = 0
    end
    for e in edges(mol) # fill dict
        key = join(sort([string(get_prop(mol, src(e), :symbol)), string(get_prop(mol, dst(e), :symbol)), string(get_prop(mol, e, :order))])) # the sorting procedure is very improtant, s.t. permutation invariance
        md[key] += 1
    end
    return md
end

function get_qm9_bondtypes(atoms, bond_level)
    acs = Combinatorics.combinations(atoms, 2)
    acstr = vcat([ac[1]*ac[2] for ac ∈ acs], [at*at for at ∈ atoms])
    acbl = Iterators.product(acstr, bond_level)
    acblstr = [ac[1]*string(ac[2]) for ac ∈ acbl]
    return join.(sort.(collect.(acblstr))) # sort alphabetically
end


function get_qm9_bondcounts()
    function extract_bonds!(bondfs, bondtypes, fpath, file; include_hydrogens=false)
        content = readdlm(fpath*file)
        natom = content[1,1]
        smiles = content[natom+4, 1]
        bondf = get_bonds_from_SMILES(bondtypes, smiles; include_hydrogens=include_hydrogens)
        push!(bondfs, bondf)
        #println(file, " is done!")
    end
    atom_types = ["H", "C", "N", "O", "F"]
    bond_levels = [1,2,3]
    bondtypes = get_qm9_bondtypes(atom_types, bond_levels) # get qm9 bondtypes, the keys of dict
    fpath = "/users/baribowo/Dataset/gdb9-14b/geometry/" # absolute path to qm9 dataset
    #exfiles = readdlm("data/qm9_error.txt") # excluded geometries
    files = readdir(fpath)
    #files = [file for file ∈ files if file ∉ exfiles] # included geom only
    bondfs = []
    @simd for file ∈ files
        @inbounds extract_bonds!(bondfs, bondtypes, fpath, file; include_hydrogens = true)
    end
    open("data/features_qm9_covalentbonds-H.json", "w") do f
        JSON.print(f, bondfs)
    end
end

function postprocess_bonds()
    atom_types = ["H", "C", "N", "O", "F"]
    bond_levels = [1,2,3]
    bondtypes = get_qm9_bondtypes(atom_types, bond_levels) # get qm9 bondtypes, the keys of dict
    bondfs = JSON.parsefile("data/features_qm9_covalentbonds-H.json")
    # get stats ∀keys:
    stat = Dict() 
    for key ∈ bondtypes
        stat[key] = 0
    end
    for bondf ∈ bondfs
        for key ∈ bondtypes
            stat[key] += bondf[key]
        end        
    end
    println(sort(collect(stat), by=x->x[2])) # get stat counts
    # get nnnz:
    count = 0;
    for key ∈ bondtypes
        if stat[key] > 0
            count += 1
        end
    end
    println(count)
    # transform to nqm9×nf matrix for fitting:
    nrow = length(bondfs); ncol = length(bondtypes)
    F = zeros(nrow, ncol)
    @simd for j ∈ eachindex(bondtypes)
        @simd for i ∈ eachindex(bondfs)
            @inbounds F[i, j] = bondfs[i][bondtypes[j]]
        end
    end
    display(F)
    writedlm("data/bond_types-H.txt", bondtypes)
    save("data/featuresmat_qm9_covalentbonds-H.jld", "data", F)
end

"""
=== DRESSED ANGLES ===
similar to dressed bonds, but it's angles
! remove_hydrogens param is removed, since now atom_types is an input
!! need to be changed, this permutation need to account for symmetry
"""
function get_angle_types_INCORRECT(atom_types, bond_levels)
    # for angles, each bond is unique, therefore just all possible combination, much larger degrees of freedom
    at_iter = Iterators.product(bond_levels, bond_levels, atom_types, atom_types, atom_types)
    ats = []
    for at ∈ at_iter
        push!(ats, join([string(at[1]), string(at[2]), at[3], at[4], at[5]]))
    end
    return ats
end


"""
get SYMMETRIC angle types given set of atoms and bonds
"""
function get_angle_types(types, degrees)
    atuple = [t^2 for t in types] # atom types equal tuple
    acomb = join.(collect(Combinatorics.combinations(types, 2))) # combination of atom types
    dtuple = [string(d)^2 for d in degrees] # degrees equal tuple
    dcomb = join.(collect(Combinatorics.combinations(degrees, 2))) # combination of degrees
    denum = join.(collect(Iterators.product(degrees, degrees))) # enumeration of degrees
    
    atdtuc = vec(join.(collect(Iterators.product(vcat(dcomb, dtuple), atuple)))) # product of atom tuple and degree tuple union combination
    acde = vec(join.(collect(Iterators.product(denum, acomb)))) # product of atom combination and degree enumeration
    chains = vcat(atdtuc, acde) # all except center of symmetry
    return symmetrize_angle.(join.(collect(Iterators.product(types, chains)))) # join with center of symmetry
end

"""
symmetrize an angle type given the string sequence (or vector)
"""
function symmetrize_angle(s)
    # if the non-center atoms are equal, then sort the bond levels, otherwise sort the non center atoms then the bonds
    if s[end] == s[end-1]
        sid = sortperm([s[2], s[3]])
        bondstr = join([s[2], s[3]][sid])
        atomstr = join([s[end-1], s[end]])
    else
        sid = sortperm([s[end-1], s[end]])
        bondstr = join([s[2], s[3]][sid])
        atomstr = join([s[end-1], s[end]][sid])
    end
    return join([s[1], bondstr, atomstr])
end


"""
get the list of angles (triplets) given an observed atom (vertex) within a molecule
use the formula: C(n_neighbours, 2) given an atom and a molecule
returns: 
returns EMPTY if there's no angles found with atom atom as the center
"""
function get_angles(mol, center_atom) # atom is the index of atom in the mol chain
    neighs = neighbors(mol, center_atom)
    n_angle = binomial(length(neighs), 2)
    # get angles:
    angles_iter = Combinatorics.combinations(neighs, 2)
    angles = zeros(Int, n_angle, 5) # triplets: (center, left, right) ∪ duplet: (left, right) bonds
    for (i,angle) ∈ enumerate(angles_iter)
        angles[i, end-1:end] = angle # atom triplets
        angles[i, 2:3] = [get_prop(mol, center_atom, angle[1], :order),
                        get_prop(mol, center_atom, angle[2], :order)]
        angles[i, 1] = center_atom

    end
    return angles
end

"""
angle_types is a dict containing vectors of each angle type
"""
function get_angles_from_SMILES(angle_types, str; include_hydrogens=false)
    # Dict output:
    dangle = Dict()
    for type ∈ angle_types
        dangle[type] = 0 # if no angles are found
    end
    # count 
    mol = smilestomol(str)
    if include_hydrogens
        add_hydrogens!(mol)
    end
    atoms = vertices(mol) 
    #list_angles = [] # for debugging
    for atom ∈ atoms
        angles = get_angles(mol, atom) # get angles and degrees from each atom 
        if !isempty(angles)
            for i ∈ axes(angles, 1)
                angle = angles[i,:]
                angle = join(string.([get_prop(mol, angle[1], :symbol),
                        angle[2], angle[3], 
                        get_prop(mol, angle[4], :symbol), 
                        get_prop(mol, angle[5], :symbol)])) # transform to string
                angle = symmetrize_angle(angle) # symmetrize angle 
                dangle[angle] += 1
                #push!(list_angles, angle)
            end
        end
    end
    return dangle #, list_angles
end

"""
computes all angles from qm9 dataset (the excluded indices are removed manually, same as bond)
"""
function main_get_qm9_angles()
    path = "../../../Dataset/gdb9-14b/geometry/" 
    files = readdir(path)
    atom_types = ["H","C","N","O","F"]; bond_levels = [1,2,3] 
    angle_types = get_angle_types(atom_types, bond_levels)
    t = @elapsed begin
        list_angles = ThreadsX.map(files) do fil
            smiles = fetch_SMILES(path*fil)
            angles = get_angles_from_SMILES(angle_types, smiles; include_hydrogens = true)
            angles
        end
    end
    display(list_angles)
    # transform to matrix:
    nrow = length(list_angles); ncol = length(angle_types)
    F = zeros(nrow, ncol)
    t_t = @elapsed begin
        @simd for j ∈ eachindex(angle_types)
            @simd for i ∈ eachindex(list_angles)
                @inbounds F[i, j] = list_angles[i][angle_types[j]]
            end
        end
    end
    display(F)
    println("elapsed = ",t, " ",t_t)
    writedlm("data/angle-H_types_qm9.txt", angle_types)
    save("data/featuresmat_angles-H_qm9.jld", "data", F)    
end


"""
remove unnecessary columns (such as zeros, H, etc), practically manual PCA
"""
function main_postprocess_angles()
    angle_types = readdlm("data/angle-H_types_qm9.txt")
    F = load("data/featuresmat_angles-H_qm9.jld", "data")
    # get statistics:
    colsum = vec(sum(F, dims=1))
    idnz = findall(colsum .> 0) # find nonzero indices
    idz = setdiff(1:size(F, 2), idnz) # actually find zero is faster
    println("(nz, z) = ", length.((idnz, idz)))
    sid = sortperm(colsum[idnz]) # get sorted ids of the nz
    display([colsum[idnz][sid] angle_types[idnz][sid]]) # sorted nz
    at_sorted = angle_types[idnz][sid]
    Hid = [] # check for "H" occurences
    for i ∈ eachindex(at_sorted) 
        if occursin("H", at_sorted[i])
            push!(Hid, i)
        end
    end
    #display([colsum[idnz][sid][Hid] at_sorted[Hid]]) # check how large the sum is before deleting the H
    # remove H and zeros columns from the matrix:
    Hid = []
    for i ∈ eachindex(angle_types)
        if occursin("H", angle_types[i])
            push!(Hid, i)
        end
    end
    #display(angle_types[Hid])
    delids = Hid ∪ idz # all of the column ids that need to be removed
    display([angle_types[delids] colsum[delids]])
    F = F[:, setdiff(1:size(F, 2), delids)]
    save("data/featuresmat_angles-H_qm9_post.jld", "data", F)
end

"""
==== DRESSED TORSION ====
"""

function symmetrize_torsion(s)
    s = collect(s) # operation on array instead of strings
    swap_bonds = false; sort_bonds = false
    if s[4] > s[end]
        temp = s[4]; s[4] = s[end]; s[end] = temp; # swap end to end
        temp = s[5]; s[5] = s[end-1]; s[end-1] = temp; # swap middle
        swap_bonds = true
    elseif s[4] == s[end]
        if s[5] == s[end-1] # symmetric or uniform
            sort_bonds = true
        elseif s[5] > s[end-1]
            temp = s[5]; s[5] = s[end-1]; s[end-1] = temp; # swap middle
            swap_bonds = true
        end
    end
    # check for bonds swapping:
    if swap_bonds # swapping following the swaps of atoms
        temp = s[2]; s[2] = s[3]; s[3] = temp; # swap bonds
    elseif sort_bonds # independent sort, only for symmetric or uniform atom
        s[2:3] = sort([s[2], s[3]])
    end
    return join(s)
end

function get_torsion_types(atom_types, bond_levels)
    at_iter = Iterators.product(bond_levels, bond_levels, bond_levels, atom_types, atom_types, atom_types, atom_types)
    ats = []
    for at ∈ at_iter
        push!(ats, join([string(at[1]), string(at[2]), string(at[3]), at[4], at[5], at[6], at[7]]))
    end
    # symmetrize types:
    ats_symm = []
    for at ∈ ats
        at_sym = symmetrize_torsion(at)
        if at_sym ∉ ats_symm
            push!(ats_symm, at_sym)
        end
    end
    return ats_symm
end

"""
get the vector of torsions (4 body) given an observed edge
"""
function get_torsions(mol, edge)
    sv = edge.src; dv = edge.dst # get source and dest
    sneigh = setdiff(neighbors(mol, sv), dv); dneigh = setdiff(neighbors(mol, dv), sv) # get the neighbors which excludes the observed edge
    #println(sneigh, dneigh)
    n_tor = length(sneigh)*length(dneigh) # number of torsions
    T = zeros(Int, n_tor, 7) # 3 edges degrees + 4 vertices
    if !isempty(sneigh) && !isempty(dneigh)
        torsions = Iterators.product(sneigh, dneigh)
        for (i,t) ∈ enumerate(torsions)
            quad = [t[1], sv, dv, t[2]] # left, source, dest, right atoms
            T[i, 4:7] = quad # vertices
            T[i, 1:3] = [get_prop(mol, quad[2], quad[3], :order), # center edge
                        get_prop(mol, quad[1], quad[2], :order), # left edge
                        get_prop(mol, quad[3], quad[4], :order)] # right edge
        end
    end
    return T
end

function get_torsions_from_SMILES(torsion_types, str; include_hydrogens=false)
    mol = smilestomol(str)
    if include_hydrogens
        add_hydrogens!(mol)
    end
    # init empty dict:
    dtorsion = Dict()
    for key ∈ torsion_types
        dtorsion[key] = 0
    end
    iter_edges = edges(mol) # get all edges
    for edge ∈ iter_edges
        T = get_torsions(mol, edge)
        if !isempty(T) # if torsions exist
            for i ∈ axes(T, 1)
                tstr = join(string.([T[i, 1], T[i, 2], T[i, 3], 
                                    get_prop(mol, T[i, 4], :symbol), 
                                    get_prop(mol, T[i, 5], :symbol), 
                                    get_prop(mol, T[i, 6], :symbol),
                                    get_prop(mol, T[i, 7], :symbol)]))
                s_symm = symmetrize_torsion(tstr)
                #println([tstr, s_symm])
                dtorsion[s_symm] += 1
            end
        end
    end
    return dtorsion
end

function main_get_qm9_torsions()
    path = "../../../Dataset/gdb9-14b/geometry/" 
    files = readdir(path)
    atom_types = ["H","C","N","O","F"]; bond_levels = [1,2,3] 
    torsion_types = get_torsion_types(atom_types, bond_levels)
    t = @elapsed begin
        list_torsions = ThreadsX.map(files) do fil
            smiles = fetch_SMILES(path*fil)
            torsions = get_torsions_from_SMILES(torsion_types, smiles; include_hydrogens=true)
            torsions
        end
    end
    # transform to matrix:
    nrow = length(list_torsions); ncol = length(torsion_types)
    F = zeros(nrow, ncol)
    t_t = @elapsed begin
        @simd for j ∈ eachindex(torsion_types)
            @simd for i ∈ eachindex(list_torsions)
                @inbounds F[i, j] = list_torsions[i][torsion_types[j]]
            end
        end
    end
    println("elapsed = ",t, " ",t_t)
    writedlm("data/torsion-H_types_qm9.txt", torsion_types)
    save("data/featuresmat_torsion-H_qm9.jld", "data", F)
end

"""
got OOM on OMP1, need to do it in VSC5
"""
function postprocess_torsion()
    F = load("data/featuresmat_torsion_qm9.jld", "data")
    tts = vec(readdlm("data/torsion_types_qm9.txt"))
    exids = readdlm("data/exids.txt")

    F = F[setdiff(1:size(F, 1), exids), :] # remove excluded molecules
    # remove H:
    idnH = []
    for (i, tt) ∈ enumerate(tts)
        if !occursin("H", tt)
            push!(idnH, i)
        end
    end
    colsum = vec(sum(F[:, idnH], dims=1))
    idnz = findall(colsum .> 0)
    writedlm("data/torsion_types_post.txt", tts[idnH][idnz]) # nonzero nonH torsion types
    save("data/featuresmat_torsion_qm9_post.jld", "data", F[:, idnH][:, idnz]) # nonzero nonH matrix
end

"""
copy paste content of the function to the terminal is probably better
"""
function test_dabat()
    str = "CC(=O)OC1=CC=CC=C1C(=O)O"
    mol = smilestomol(str)
    e = collect(edges(mol))[3]
    # each row is a sequence of a torsion:
    T = get_torsions(mol, e)
    display(T)
    display(string.(map(x->get_prop(mol, x, :symbol), T[:, 4:7])))
    atom_types = ["H","C","N","O","F"]; bond_levels = [1,2,3];
    torsion_types = get_torsion_types(atom_types, bond_levels)
    display(torsion_types)
    T = get_torsions_from_SMILES(torsion_types, str)
    display(T["112CCCO"])
end

"""
extract all smiles excluding the excludedd ones and save to one file
"""
function main_extract_smiles()
    exs = Int.(readdlm("data/exids.txt")[:])
    path = "../../../Dataset/gdb9-14b/geometry/" 
    files = readdir(path)
    smiless = []
    for f ∈ files
        content = readdlm(path*f)
        natom = content[1,1]
        smiles = content[Int(natom)+4, 1]
        push!(smiless, smiles)
    end
    smiless = smiless[setdiff(1:length(smiless),exs)]
    return smiless
end

"""
== Hybridization of dressed stuffs ==
"""

"""
generate all possible hybrid types of each atom type
"""
function generate_hybrid_types(atom_types, hybrid_types)
    hiter = Iterators.product(atom_types, hybrid_types)
    hybrids = []
    for h ∈ hiter
        hstr = join([h[1], h[2]])
        push!(hybrids, hstr)
    end
    return hybrids
end

"""
count the hybrids exist in a molecule (SMILES representation) given atom × hybrid types
"""
function count_hybrids(ahtypes, smiles)
    # generate empty dict:
    dh = Dict()
    for a ∈ ahtypes
        dh[a] = 0
    end
    # count hybrids:
    mol = smilestomol(smiles)
    atoms = atom_symbol(mol)
    hybrids = hybridization(mol)
    for i ∈ eachindex(atoms)
        ahstr = join([atoms[i], hybrids[i]])
        dh[ahstr] += 1
    end
    return dh
end

function test_hybrid()
    path = "../../../Dataset/gdb9-14b/geometry/" 
    files = readdir(path)
    ahtypes = generate_hybrid_types(["C", "N", "O", "F"], [:none, :sp, :sp2, :sp3])
    for f ∈ files[[1, end]]
        smiles = fetch_SMILES(path*f)
        println(smiles)
        println(count_hybrids(ahtypes, smiles))
    end
end

"""
get the hybrids of the whole qm9 dataset
"""
function main_get_qm9_hybrids()
    path = "../../../Dataset/gdb9-14b/geometry/" 
    files = readdir(path)
    # exlcude H from the matrix later, since it will be substituted with the atomref_features[:,"H"] instead later:
    ahtypes = generate_hybrid_types(["H", "C", "N", "O", "F"], [:none, :sp, :sp2, :sp3])
    t = @elapsed begin
        list_hybrids = ThreadsX.map(files) do fil
            smiles = fetch_SMILES(path*fil)
            hybrids = count_hybrids(ahtypes, smiles)
            hybrids
        end
    end
    println("hybrid FE done in ", t)
    # transfomr to matrix:
    F = zeros(length(list_hybrids), length(ahtypes))
    @simd for j ∈ axes(F, 2)
        @simd for i ∈ axes(F, 1)
            @inbounds F[i,j] = list_hybrids[i][ahtypes[j]]
        end
    end
    writedlm("data/atom_types_hybrid.txt", ahtypes)
    save("data/featuresmat_atomhybrid_qm9.jld", "data", F)
end

function main_post_hybrids()
    # slice the rows of dressed hybrid atom:
    hts = vec(readdlm("data/atom_types_hybrid.txt"))
    exids = vec(readdlm("data/exids.txt"))
    Fh = load("data/featuresmat_atomhybrid_qm9.jld", "data")
    Fh = Fh[setdiff(1:size(Fh, 1), exids), :]
    # check if the sums are correct:
    println(sum(vec(sum(Fh, dims=1))[[1,6,11,16] .+ 1])) # [1,6,11,16] := the indices of "H", +1 := H+1 = C, and so on...
    # exclude the H and zeros:
    Hids = [1,6,11,16] # H
    colsum = vec(sum(Fh, dims=1))
    zeroids = findall(colsum .== 0.) # zeros
    excolids = Hids ∪ zeroids # union
    Fh = Fh[:, setdiff(1:length(hts), excolids)] # slice columns
    # concatenate H with dressed atom:
    Fd = load("data/atomref_features.jld", "data")
    Fh = hcat(Fd[:, 1], Fh)
    
    writedlm("data/atom_types_hybrid_post.txt", hts[setdiff(1:length(hts), excolids)]) # save postprocessed types
    save("data/featuresmat_atomhybrid_qm9_post.jld", "data", Fh)
end

"""
=== Bonds affected by distances ===
"""

"""
computes a symmetric distance matrix given a matrix of coordinates (row = atom)
"""
function get_distance_matrix(X)
    natom = size(X, 1)
    D = zeros(Float64, natom, natom)
    for j ∈ axes(X, 1)
        for i ∈ axes(X, 1)
            D[i,j] = norm(X[i,:] - X[j,:])
        end
    end
    return D
end

function main_get_distance_matrices()
    dataset = load("data/qm9_dataset.jld", "data")
    Ds = Vector{Matrix{Float64}}() # a vector of matrices
    coords = map(d -> d["coordinates"], dataset) 
    for d ∈ coords
        push!(Ds, get_distance_matrix(d))
    end
    save("data/distance_matrices_qm9.jld", "data", Ds)
end

"""
bond type extractor given a geometry file
"""
function extract_bondtypes!(bondfs, bondtypes, fpath, file; include_hydrogens=false)
    content = readdlm(fpath*file)
    natom = content[1,1]
    smiles = content[natom+4, 1]
    bondf = get_bonds_from_SMILES(bondtypes, smiles; include_hydrogens=include_hydrogens)
    push!(bondfs, bondf)
    #println(file, " is done!")
end

function get_bonds_from_SMILES(bondtypes, str; include_hydrogens=false)
    mol = smilestomol(str)
    if include_hydrogens
        add_hydrogens!(mol)
    end
    md = Dict()
    for key ∈ bondtypes # init dict
        md[key] = 0
    end
    for e in edges(mol) # fill dict
        key = join(sort([string(get_prop(mol, src(e), :symbol)), string(get_prop(mol, dst(e), :symbol)), string(get_prop(mol, e, :order))])) # the sorting procedure is very improtant, s.t. permutation invariance
        md[key] += 1
    end
    return md
end

"""
extract smiles string given a geometry file
"""
function extract_SMILES(filepath)
    content = readdlm(filepath)
    natom = content[1,1]
    smiles = content[natom+4, 1]
    return smiles
end

"""
================
================
Higher resolution feature extractors:
================
================
"""

function get_self_interaction(Z)
    return .5*Z^2.4
end

function get_pair_interaction(Z1,Z2,R1,R2)
    return Z1*Z2/norm(R1-R2)
end

"""
generates global indexer of the bag positions (per dataset)
order: H, C, N, O, F, HH, HC, ...,CC,...FF (sorted lexicographically actually)
!! need to find out a way s.t the dict not replace each other keys --> using merge
!! need to also enumerate the same-type interactions (NOT self interaction), e.g., C₁-C₂, O₁-O₃, .... --> enumerate the identical pairs first
"""
function generate_bob_indexer(;bsizes = Dict("H"=>20, "C"=>9, "N"=>7, "O"=>5, "F"=>6))
    # generate sizes:
    s = values(bsizes)
    # generate self interactinos indices:
    dbags = Dict()
    idx = 1
    for k ∈ keys(bsizes)
        first = idx; last = idx+bsizes[k]-1 
        dbags[k*"_self"] = [first, last]
        idx = last+1 
    end
    #display(dbags)
    # generate pair (identical) interaction indices:
    ks = string.(keys(bsizes))
    for k ∈ sort(ks)
        first = idx; last = idx + bsizes[k]^2 - 1 
        #display([k, dbags[k*"_self"], first, last])
        dbags[k] = Dict(k=>[first, last])
        idx = last+1
    end
    #display(dbags)
    # non-identical:
    kcombs = sort(collect(Combinatorics.combinations(sort(ks), 2))) # double sort for symmetry
    #display(kcombs)
    for kc ∈ kcombs
        #display(kc)
        first = idx; last = idx + bsizes[kc[1]]*bsizes[kc[2]] - 1 # index continues from the self interactions
        merge!(dbags[kc[1]], Dict(kc[2] => [first, last]))
        idx = last+1
    end
    return dbags, idx-1 # returns the dict and the size of the vector
end

"""
main extractor of BoB
"""
function main_extract_BOB()
    dataset = load("data/dataset.jld", "data")
    X = generate_bobs(dataset) # no change to the atom sizes
    save("data/BOB.jld","data",X)
end

"""
params:
    - dmol, a dictionary containing atleast {"coordinates" ∈ R^3, "atoms" := list of atom types}
    - dcharges, dict that translates atom_type -> nuclear_charges
    - dindexer, dict with at least {"atom_self" ∀atom, product(atoms)}
    - maxsize = the size of the whole bag vector
returns a vector R^maxsize
"""
function generate_bob(dmol, dcharges, dindexer, maxsize) # for each data (molecule)
    dindcp = deepcopy(dindexer) # copy indexer dict, inefficient but for now this works
    X = zeros(maxsize) # initialize whole bag vector 
    # compute self interactions:
    for atom ∈ dmol["atoms"]
        Z = dcharges[atom]
        X[dindcp[atom*"_self"][1]] = get_self_interaction(Z)
        dindcp[atom*"_self"][1] += 1 # increment the "first" cell for the next atom
    end
    # compute pair interactions:
    atoms = dmol["atoms"]
    aidx = eachindex(atoms)
    cidx = Combinatorics.combinations(aidx, 2)
    for cid ∈ cidx
        Z1 = dcharges[atoms[cid[1]]]; Z2 = dcharges[atoms[cid[2]]] # get pair charges
        R1 = dmol["coordinates"][cid[1],:]; R2 = dmol["coordinates"][cid[2],:] # get pair coords
        sortkey = sort([atoms[cid[1]], atoms[cid[2]]]) # since dindcp dictionary is sorted
        X[dindcp[sortkey[1]][sortkey[2]][1]] = get_pair_interaction(Z1,Z2,R1,R2) # compute pair interaction
        dindcp[sortkey[1]][sortkey[2]][1] += 1 # increment pair index
        #display([dindcp[sortkey[1]][sortkey[2]][1], sortkey, cid, atoms[cid[1]], atoms[cid[2]], Z1, Z2, R1, R2])
    end
    return X
end

"""
computes the BoB for whole dataset
params:
    - mols, [dict("atoms", "coordinates")] (atleast)
    - bsizes, Dict containing the max sizes of each atom type
"""
function generate_bobs(mols; bsizes = Dict("H"=>20, "C"=>9, "N"=>7, "O"=>5, "F"=>6)) # for whole dataset
    dcharges = Dict("H"=>1, "C"=>6, "N"=>7, "O"=>8, "F"=>9) # atom -> charge translator
    dindexer, maxsize = generate_bob_indexer(;bsizes) # generate indexer
    # init zeros matrix:
    nrow = length(mols); ncol = maxsize 
    X = zeros(nrow, ncol)
    for i ∈ 1:nrow # can be threadized by ThreadsX but unnecessary :p
        X[i, :] = generate_bob(mols[i], dcharges, dindexer, maxsize)
    end
    return X
end

"""
==============
feature extractors mainly calling from py
==============
"""

"""
extracts cMBDF (change of cMBDF version can be done in moldesc_min.py)
"""
function extract_CMBDF(;version = "joblib", postfix="joblib_020224")
    dataset = load("data/qm9_dataset.jld", "data")
    pushfirst!(pyimport("sys")."path", "") # load all py files in current directory
    moldesc_min = pyimport("moldesc_min") # import moldesc_min
    reps = moldesc_min.extract_MBDF([1,2,3], version) # extract (with added hyperparameters later)
    f = [reps[i,:,:] for i in axes(reps, 1)] # transform to vector of matrices
    # exclude some ids here since normalization might affect the numerics if done pre-extraction:
    idsel = setdiff(eachindex(f), vec(Int.(readdlm("data/exids.txt")))) 
    f = f[idsel]
    # remove zeros:
    for i ∈ eachindex(dataset)
        n_atom = dataset[i]["n_atom"]
        f[i] = f[i][1:n_atom, :]
    end
    save("data/CMBDF_"*postfix*".jld", "data", f)
end


"""
==============
morse potential as feature, r is the only non hyperparameter
==============
"""
function morse_pot(r, D, a, r0, s)
    return D*(exp(-2*a*(r-r0)) - 2*exp(-a*(r-r0))) + s # additional shift constant s for "pure" fitting
end

function main_morse_pot()
    # simple example of getting distances given graph info (matching numbers -- as prof.edelman says)
    # get the ids foreach pair then compute the whole thing:
    geopath = "/users/baribowo/Dataset/gdb9-14b/geometry/"; geoms = readdir(geopath)
    dataset = load("data/qm9_dataset.jld", "data")
    exids = Int.(vec(readdlm("data/exids.txt")))
    coords = map(d ->d["coordinates"], dataset)
    D = load("data/distance_matrices_qm9.jld", "data")
    bondtypes = readdlm("data/bond_types-H_nz.txt") # nonzero bondtypes that exist, determines the order
    
    # extract smiles:
    geoms = geoms[setdiff(1:length(geoms), exids)]
    fpaths = [geopath*geom for geom ∈ geoms]
    smiless = ThreadsX.map(fpaths) do fpath
        extract_SMILES(fpath) 
    end
    # test one mol:
    mol = smilestomol("C")
    add_hydrogens!(mol)
    es = edges(mol)
    for e ∈ es
        pairid = [src(e), dst(e)]
        btype = join(sort([string(get_prop(mol, src(e), :symbol)), string(get_prop(mol, dst(e), :symbol)), string(get_prop(mol, e, :order))]))
        pairids[btype] = pairid
        println(e," ",D[1][src(e), dst(e)]) # how to query distance
    end
    display(D[1])

    # get the ids, computed once per dataset:
    Ps = Vector{Matrix{Float64}}() # this should be global -> a vector of matrices ∈ Z^{n_pair,2}
    # generate dict, test with mol #(1,2):
    for i ∈ eachindex(dataset)[1:2]
        # this will be inside a function:
        P = Dict()
        for (i,t) ∈ enumerate(bondtypes)
            P[t] = []
        end
        
        push!(Ps, P)
    end
    


    d_ct = Dict() # generate the index translator
    for (i,t) ∈ enumerate(bondtypes)
        d_ct[t] = i
    end

    # compute the
end

