"""
!!! TRANSFORM THE COORDS TO HARTREE!
"""


using DelimitedFiles, DataStructures, HDF5, JLD, BenchmarkTools, Printf


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
function getSOAP()
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

function table_results(foldername)
    sheader = ["n_data", "n_af", "n_mf", "n_basis", "num_centers", "ft_sos", "ft_bin"]
    rheader = ["Nqm9", "nK", "nU", "n_feature", "n_basis", "MAE", "RMSD", "max(MAD)", "t_ab", "t_ls", "t_batch"]
    outheader = ["nK", "nU", "n_af", "n_mf", "n_basis", "MAE", "ft_sos", "ft_bin", "t_solver", "t_pred"]
    setup_info = readdlm("data/$foldername/setup_info.txt", '\t', '\n')
    res_info = readdlm("result/$foldername/err_$foldername.txt", '\t', '\n')
    #display(setup_info)
    #display(res_info)
    out = Dict()
    out["nK"] = res_info[:, 2]
    out["nU"] = res_info[:, 3]
    out["n_af"] = setup_info[:, 2]
    out["n_mf"] = res_info[:, 4]
    out["n_basis"] = res_info[:, 5]
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
