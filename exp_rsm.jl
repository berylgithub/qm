include("RoSemi.jl")
include("expdriver.jl")
include("utils.jl")

using DelimitedFiles
using Random, Combinatorics
using MLBase, Hyperopt

"""
converts distance to r^k/(r^k+r0^k), k=1,2,3, r0 ≈ req
"""
function rdist(r,r0;c=1)
    return r^c/(r^c + r0^c)
end

"""
converts distance to e^{-r/r0}
"""
function edist(r,r0)
    return exp(-r/r0)
end



"""
rosemi wrapper fitter, can choose either with kfold or usequence (if kfold = true then pcs isnt used)   
    - F = feature vector (or matrix)
    - E = target energy vector
    - kfold = use kfold or usequence
    - k = number of folds
    - pcs = percentage of centers
    - ptr = percentage of trainings
"""
function rosemi_fitter(F, E, folds; pcs = 0.8, ptr = 0.5, n_basis=4, λ = 0., force=true)
    ndata = length(E)
    #folds = collect(Kfold(ndata, k)) # for rosemi, these guys are centers, not "training set"
    #println(folds)
    MAEs = []; RMSEs = []; RMSDs = []; t_lss = []; t_preds = []
    for (i,fold) in enumerate(folds)
        # fitter:
        ϕ, dϕ = extract_bspline_df(F', n_basis; flatten=true, sparsemat=true)
        #fold = shuffle(fold); 
        centers = fold; lenctr = length(centers) # the center ids
        trids = fold[1:Int(round(ptr*lenctr))] # train ids (K)
        uids = setdiff(fold, trids) # unsupervised ids (U)
        tsids = setdiff(1:ndata, fold)
        D = fcenterdist(F, centers) .+ λ # temp fix for numericals stability
        bsize = max(1, Int(round(0.25*length(tsids)))) # to avoid bsize=0
        MAE, RMSE, RMSD, t_ls, t_pred = fitter(F', E, D, ϕ, dϕ, trids, centers, uids, tsids, size(F, 2), "test", bsize, 900, get_rmse=true, force=force)    
        MAE = MAE/627.503 ## MAE in energy input's unit
        # store results:
        push!(MAEs, MAE); push!(RMSEs, RMSE); push!(RMSDs, RMSD); push!(t_lss, t_ls); push!(t_preds, t_pred); 
    end
    return MAEs, RMSEs, RMSDs, t_lss, t_preds 
end


"""
rerun of HxOy using ROSEMi
"""
function main_rosemi_hxoy(;force=true, c=1, n_basis=4, ptr=0.5)
    Random.seed!(603)
    # pair HxOy fitting:
    data = load("data/smallmol/hxoy_data_req.jld", "data") # load hxoy
    # do fitting for each dataset:
    # for each dataset split kfold
    # possibly rerun with other models?
    ld_res = []
    for i ∈ eachindex(data)[3:3]
        # extract data:
        d = data[i]
        req = d["req"]
        F = rdist.(d["R"], req, c=c) #edist.(d["R"], req) # convert distance features
        E = d["V"]
        println([d["mol"], d["author"], d["state"]])
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, force=force)
        display([MAEs, RMSEs, RMSDs, t_lss, t_preds ])
        println("RMSE = ", RMSEs)
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"]; d_res["author"] = d["author"]; d_res["state"] = d["state"]; 
        display(d_res)
        push!(ld_res, d_res)
    end
    #save("result/hdrsm_[$force].jld", "data", ld_res)
end


"""
rerun of Hn, 3 ≤ n ≤ 5 molecules using ROSEMI
"""
function main_rosemi_hn(;force=true)
    Random.seed!(603)
    data = load("data/smallmol/hn_data.jld", "data")
    ld_res = []
    for i ∈ eachindex(data)
        λ = 0.
        d = data[i]; F = d["R"]; E = d["V"]
        if d["mol"] == "H5" # for H5, add "regularizer"
            λ = 1e-8
        end
        println(d["mol"])
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=4, ptr=0.5, λ = λ, force=force) # λ = 1e-8
        display([MAEs, RMSEs, RMSDs, t_lss, t_preds])
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"];
        push!(ld_res, d_res)
    end
    save("result/hn_rosemi_rerun_[$force].jld", "data", ld_res) #save("result/h5_rosemi_rerun_unstable.jld", "data", ld_res) 
end

"""
-----------
hyperparamopt routines
-----------
"""

"""
objective function for hyperparam opt 
"""
function rosemi_fobj(R, E, req, folds; force=true, c=1, n_basis=4, ptr=0.5)
    F = rdist.(R, req; c=c)
    MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, force=force)
    #display(RMSEs)
    return mean(RMSEs) # would mean() be a better metric here? or min() is preferrable?
end


function main_hpopt_rsm()
    Random.seed!(603)
    # pair HxOy fitting:
    data = load("data/smallmol/hxoy_data_req.jld", "data") # load hxoy
    # do fitting for each dataset:
    # for each dataset split kfold
    # possibly rerun with other models?
    ld_res = []
    is = setdiff(eachindex(data), 10) # exclude the large H2 data, since it might be slow
    for i ∈ is
        d = data[i]
        req = d["req"]
        println(d["mol"])
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        t = @elapsed begin
            ho = @thyperopt for i=500, # hpspace = 2*3*10*9 = 540
                    sampler = RandomSampler(),
                    force=[false,true],
                    c=[1,2,3],
                    n_basis = collect(1:10),
                    ptr = LinRange(0.1, 0.9, 9)
                fobj = rosemi_fobj(d["R"], d["V"], req, folds; force=force, c=c, n_basis=n_basis, ptr=ptr)
            end
        end    
        best_params, min_f = ho.minimizer, ho.minimum
        display(best_params)
        display(min_f)
        # save using txt to avoid parallelization crash:
        di = vcat(d["mol"], d["author"], min_f, t, collect(best_params))
        push!(ld_res, di)
        out = reduce(vcat,permutedims.(ld_res))
        writedlm("data/smallmol/hpopt_hxoy_rsm.text", out)
    end
end
