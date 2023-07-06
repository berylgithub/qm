"""
!!! TRANSFORM THE COORDS TO HARTREE!

The collection of functions for (d)ata (p)reparation
"""


using DelimitedFiles, DataStructures, JSON, JLD, BenchmarkTools, Printf
using Graphs, MolecularGraph, Combinatorics, SparseArrays # stuffs for ΔML
using LinearAlgebra


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
(prototype) get the bond order given smiles string of a molecule
returns dict of bondtype => count
"""
function get_bonds_from_SMILES(bondtypes, str; remove_hydrogens=true)
    mol = smilestomol(str)
    if remove_hydrogens
        remove_hydrogens!(mol)
    end
    md = Dict()
    for key ∈ bondtypes # init dict
        md[key] = 0
    end
    for e in edges(mol) # fill dict
        key = join(sort([string(get_prop(mol, src(e), :symbol)), string(get_prop(mol, dst(e), :symbol)), string(get_prop(mol, e, :order))]))
        md[key] += 1
    end
    return md
end

function get_qm9_bondtypes(;remove_hydrogens=true)
    atoms = ["H", "C", "N", "O", "F"]
    if remove_hydrogens
        atoms = atoms[2:end]
    end
    bond_level = [1,2,3]
    acs = Combinatorics.combinations(atoms, 2)
    acstr = vcat([ac[1]*ac[2] for ac ∈ acs], [at*at for at ∈ atoms])
    acbl = Iterators.product(acstr, bond_level)
    acblstr = [ac[1]*string(ac[2]) for ac ∈ acbl]
    return join.(sort.(collect.(acblstr))) # sort alphabetically
end

function get_qm9_bondcounts()
    function extract_bonds!(bondfs, bondtypes, fpath, file; remove_hydrogens=true)
        content = readdlm(fpath*file)
        natom = content[1,1]
        smiles = content[natom+4, 1]
        bondf = get_bonds_from_SMILES(bondtypes, smiles; remove_hydrogens=remove_hydrogens)
        push!(bondfs, bondf)
        #println(file, " is done!")
    end
    remove_hydrogens = true
    bondtypes = get_qm9_bondtypes(;remove_hydrogens = remove_hydrogens) # get qm9 bondtypes, the keys of dict
    fpath = "/users/baribowo/Dataset/gdb9-14b/geometry/" # absolute path to qm9 dataset
    #exfiles = readdlm("data/qm9_error.txt") # excluded geometries
    files = readdir(fpath)
    #files = [file for file ∈ files if file ∉ exfiles] # included geom only
    bondfs = []
    @simd for file ∈ files
        @inbounds extract_bonds!(bondfs, bondtypes, fpath, file; remove_hydrogens = remove_hydrogens)
    end
    open("data/features_qm9_covalentbonds.json", "w") do f
        JSON.print(f, bondfs)
    end
end

function postprocess_bonds()
    bondtypes = get_qm9_bondtypes(;remove_hydrogens=true) # get qm9 bondtypes, the keys of dict
    bondfs = JSON.parsefile("data/features_qm9_covalentbonds.json")
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
    #writedlm("data/featuresmat_qm9_covalentbonds.txt", F)
    save("data/featuresmat_qm9_covalentbonds.jld", "data", F)
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
remove the molids of uncharacterized ∪ non converged geos
"""
function feature_slicer()
    slicer = vec(Int.(readdlm("data/exids.txt")))
    #feature_paths = ["data/qm9_dataset.jld", "data/FCHL19.jld", "data/SOAP.jld", "data/SOAP.jld"] 
    #feature_paths = ["data/atomref_features.jld", "data/featuresmat_qm9_covalentbonds.jld"] # all dataset then features
    feature_paths = ["data/ACSF.jld"]
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