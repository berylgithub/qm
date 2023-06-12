"""
!!! TRANSFORM THE COORDS TO HARTREE!

The collection of functions for (d)ata (p)reparation
"""


using DelimitedFiles, DataStructures, JLD, BenchmarkTools, Printf


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
    files = readdir("data/qm9/")
    dir = "data/qm9/"
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

function generate_charges_distances()
    # readpath and prep info
    geopath = "../../Dataset/zaspel_supp/geometry/" # geometry  path, can be used for other dataset too as long as the format is the same
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
    display(moldata)
    save("deltaML/data/zaspel_ncd.jld", "data", moldata)
end