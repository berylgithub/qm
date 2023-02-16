include("alouEt.jl")

function caller_ds()
#=     # FEEATURE EXTRACTION:
    foldername = "exp_reduced_energy"
    # ACSF:
    nafs = [40, 30, 20] 
    nmfs = [40, 30, 20]
    println("ACSF")
    for i ∈ eachindex(nafs)
        for j ∈ eachindex(nmfs)
            nmf = nmfs[j]
            if nmf >= nafs[i] # the num of mol feature must be less than atom f
                nmf = nafs[i] - rand(1:5, 1)[1]
            end
            println(nafs[i]," ",nmf)
            data_setup(foldername, nafs[i], nmf, 3, 300, "data/qm9_dataset_old.jld", "data/ACSF.jld", "ACSF"; 
                        save_global_centers = true, num_center_sets = 5)
            GC.gc()
        end
    end
    # SOAP:
    nafs = [75, 50, 25]
    nmfs = [75, 50, 25]
    println("SOAP")
    for i ∈ eachindex(nafs)
        for j ∈ eachindex(nmfs)
            nmf = nmfs[j]
            if nmf >= nafs[i] # the num of mol feature must be less than atom f
                nmf = nafs[i] - rand(1:5, 1)[1]
            end
            println(nafs[i]," ",nmf)
            data_setup(foldername, nafs[i], nmf, 3, 300, "data/qm9_dataset_old.jld", "data/SOAP.jld", "SOAP"; 
                        save_global_centers = true, num_center_sets = 5)
            GC.gc()
        end
    end =#

    # ATOM FITTING:
    uids = readdlm("data/centers.txt")[:, 1]
    kids = readdlm("data/centers.txt")[:, 2]
    center_sets = Int.(readdlm("data/centers.txt")[:, 3:end]) # a matrix (n_sets, n_centers) ∈ Int
    display(center_sets)
    for i ∈ eachindex(uids)
        println(uids[i]," ",kids[i])
        fit_atom("exp_reduced_energy", "data/qm9_dataset_old.jld", "data/atomref_features.jld";
              center_ids=center_sets[i, :], uid=uids[i], kid=kids[i])
        GC.gc()
    end
end


function caller_fit()
    # get the 10 best:
    atom_info = readdlm("result/exp_reduced_energy/atomref_info.txt")[4:end, :] # the first 3 is excluded since theyr noise from prev exps
    MAVs = Vector{Float64}(atom_info[:, 3]) # 3:=training MAE, 4:=training+testing MAE
    sid = sortperm(MAVs) # indices from sorted MAV
    best10 = atom_info[sid[1:10], :] # 10 best
    display(best10)
    # compute total atomic energy:
    E_atom = Matrix{Float64}(best10[:, end-4:end]) # 5 length vector
    E = vec(readdlm("data/energies.txt"))
    f_atomref = load("data/atomref_features.jld", "data") # n × 5 matrix
    ds = readdlm("data/exp_reduced_energy/setup_info.txt")
    ds_uids = ds[:, 1] # vector of uids
    centers_info = readdlm("data/centers.txt")
    centers_uids = centers_info[:, 1]
    centers_kids = centers_info[:, 2]
    models = ["ROSEMI", "KRR", "NN", "LLS", "GAK"] # for model loop, |ds| × |models| = 50 exps
    for k ∈ axes(best10, 1) # loop k foreach found best MAV
        E_null = f_atomref*E_atom[k,:] # Ax, where A := matrix of number of atoms, x = atomic energies
        #MAV = mean(abs.(E - E_null))*627.5
        id_look = best10[k, 1] # the uid that we want to find
        id_k_look = best10[k, 2] # the uid of the center that we want to find
        # look for the correct centers:
        found_ids = []
        for i ∈ eachindex(centers_uids)
            if id_look == centers_uids[i]
                push!(found_ids, i)
            end 
        end
        #display(centers_info[found_ids, :])
        # get the correct center id:
        found_idx = nothing
        for i ∈ eachindex(found_ids)
            if id_k_look == centers_kids[found_ids[i]]
                found_idx = i
                break
            end
        end
        #display(centers_info[found_ids[found_idx], :])
        center = centers_info[found_ids[found_idx], 3:end]
        #display(center)
        # look for the correct hyperparameters:
        found_idx = nothing
        for i ∈ eachindex(ds_uids)
            if id_look == ds_uids[i]
                found_idx = i
                break
            end
        end
        featname, naf, nmf = ds[found_idx, [4, 5, 6]] # the data that we want to find
        featfile = ""
        if featname == "SOAP"
            featfile = "data/SOAP.jld"
        elseif featname == "ACSF"
            featfile = "data/ACSF.jld"
        end
        uid = id_look
        kid = id_look*"_"*best10[k, 2]
        #println([uid," ",kid])
        println([k, featname, naf, nmf, kid])
        # test one ds and fit
        data_setup("exp_reduced_energy", naf, nmf, 3, 300, "data/qm9_dataset_old.jld", featfile, featname)
        GC.gc()
        # loop the model:
        for model ∈ models
            fit_🌹_and_atom("exp_reduced_energy", "data/qm9_dataset_old.jld";
                            model=model, E_atom=E_null, center_ids=center, uid=uid, kid=kid)
            GC.gc()
        end
    end
end


"""
big main function here, to tune hyperparameters by DFO
"""
function hyperparamopt(;init=false)
    # test "threading" loop:
    # initial fitting, initialize params and funs, replace with actual fitting:
    if init
        uid = replace(string(Dates.now()), ":" => ".")
        #x = [.5, .5, 3, 1, 0, 0, 0, 6, 32.0]  # init using best params
        x = [-100., -100.]
        f = main_obj(x)
        data = Matrix{Any}(undef, 1,3)
        data[1,1] = uid; data[1,2:end] = x 
        writestringline(string.(vcat(uid, x)), "params.txt")
        writestringline(string.(vcat(uid, f)), "fun.txt")
        println("init done")
    else
        data = readdlm("params.txt")
        x = data[1,2:end]
        f = sum(x .^ 2)
        # write result to file:
        uid = replace(string(Dates.now()), ":" => ".")
        writestringline(string.(vcat(uid, f)), "fun.txt")
    end
    while true
        newdata = readdlm("params.txt")
        if data[1,1] != newdata[1,1]
            data = newdata
            println("new incoming data ", data)
            # do fitting:
            x = data[1,2:end]
            f = sum(x .^ 2)
            # write result to file:
            uid = replace(string(Dates.now()), ":" => ".")
            writestringline(string.(vcat(uid, f)), "fun.txt")
        end
        sleep(0.3)
    end
end

"""
processes the raw parameters from mintry, then compute the MAE from the given parameters
par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
par_fit_atom = [center_ids] # center_ids = 0 → use 
par_fit = [model, cσ]
params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
           1       2   3           4           5               6               7       8       9
naf, nmf in percentage, e.g., .5 -> .5*max(naf("ACSF"))
feature name: 1=ACSF, 2=SOAP, 3=FCHL
model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
"""
function main_obj(x)
    # process params:
    fnaf = Dict() # determines max n_af
    fnaf[1] = 51 # ACSF
    fnaf[2] = 165 # SOAP
    fnaf[3] = 140 # FCHL
    max_naf = fnaf[x[4]]
    
    MAE = nothing 
    data_setup(foldername, nafs[i], nmf, 3, 300, "data/qm9_dataset_old.jld", "data/ACSF.jld", "ACSF"; 
        save_global_centers = true, num_center_sets = 5)
    fit_atom(foldername, file_dataset, file_atomref_features; center_ids = [], tlimit = 900, uid = "", kid = "")
    GC.gc() # always gc after each run
    fit_🌹_and_atom(foldername, file_dataset; 
        bsize = 1000, tlimit = 900, model = "ROSEMI", 
        E_atom = [], cσ = 2.0 * (2.0 ^ 5) ^ 2, scaler = 2.0 * (2.0 ^ 5) ^ 2, 
        center_ids = [], uid = "", kid = "")
    par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
    par_fit_atom = [center_ids] # center_ids = 0 → use new center, otherwise use precomputed centers
    par_fit = [model, cσ]
    GC.gc() # always gc after each run

    return MAE
end

# script to write string given a vector{string}
function writestringline(strinput, filename; mode="w")
    open(filename, mode) do io
        str = ""
        for s ∈ strinput
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
end

# script to write to latex table, given a Matrix{Any}
function writelatextable(table, filename)
    open(filename, "w") do io
        for i ∈ axes(table, 1)
            str = ""
            for j ∈ axes(table, 2)
                str *= string(table[i, j])*"\t"*"& "
            end
            str = str[1:end-2]
            str *= raw"\\ \hline"*"\n"
            print(io, str)
        end
    end
end