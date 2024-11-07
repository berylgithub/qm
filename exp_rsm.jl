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
function main_rosemi_hn(;force=true, c=1, n_basis=5, ptr=0.6) # the hyperparams are from the optimized H2 Kolos
    Random.seed!(603)
    data = load("data/smallmol/hn_data.jld", "data")
    ld_res = []
    for i ∈ setdiff(eachindex(data), [2,3]) # skip partial H4
        λ = 0.
        d = data[i]; F = rdist.(d["R"], 1.401, c=c); # req of H2  
        E = d["V"]
        if d["mol"] == "H5" # for H5, add "regularizer"
            λ = 1e-8
        end
        println(d["mol"])
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, λ = λ, force=force) 
        display([MAEs, RMSEs, RMSDs, t_lss, t_preds])
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"];
        push!(ld_res, d_res)
    end
    timestamp = replace(replace(string(now()), "-"=>""), ":"=>"")[1:end-4] # exclude ms string
    save("result/hn_rosemi_rerun_$timestamp.jld", "data", ld_res) #save("result/h5_rosemi_rerun_unstable.jld", "data", ld_res) 
end

"""
-----------
hyperparamopt routines
-----------
"""

"""
objective function for hyperparam opt 
"""
function rosemi_fobj(R, E, req, folds; force=true, c=1, n_basis=4, ptr=0.5, λ = 0.)
    F = rdist.(R, req; c=c)
    MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, force=force, λ = λ)
    #display(RMSEs)
    return mean(RMSEs) # would mean() be a better metric here? or min() is preferrable?
end


"""
call this by calling:

strnow = replace(replace(string(now()), "-"=>""), ":"=>"")[1:end-4] # current datetime in string format
main_hpopt_rsm(load("data/smallmol/hxoy_data_req.jld", "data")[setdiff(1:15, 10)]; iters=540, simid="hxoy_"*strnow, save_folds=true)
"""
function main_hpopt_rsm(data; iters=100, simid="", save_folds=false)
    Random.seed!(603)
    #data = load("data/smallmol/hxoy_data_req.jld", "data") # load hxoy
    # do fitting for each dataset:
    # for each dataset split kfold
    # possibly rerun with other models?
    ld_res = []
    foldss = [] # k-fold storage, in case random gives different numbers (in different machine could happen)
    for j ∈ eachindex(data)
        λ = 0.
        d = data[j]
        req = d["req"]
        println(d["mol"])
        if d["mol"] ∈ ["H4", "H5"] # λ > 0 if H4 or H5 for numerical stability
            println("λ is activated")
            λ = 1e-8
        end
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        if save_folds
            fd = Dict("mol" => d["mol"], "folds"=>folds)
            push!(foldss, fd)
        end
        t = @elapsed begin
            ho = @thyperopt for i=iters, # hpspace = 2*3*10*9 = 540
                    sampler = RandomSampler(),
                    #force=[false,true],
                    c=[1,2,3],
                    n_basis = collect(1:10),
                    ptr = LinRange(0.1, 0.9, 9)
                fobj = rosemi_fobj(d["R"], d["V"], req, folds; force=false, c=c, n_basis=n_basis, ptr=ptr, λ = λ)
            end
        end    
        best_params, min_f = ho.minimizer, ho.minimum
        display(best_params)
        display(min_f)
        # save using txt to avoid parallelization crash:
        dj = vcat(d["mol"], d["author"], min_f, t, collect(best_params))
        push!(ld_res, dj)
        out = reduce(vcat,permutedims.(ld_res))
        writedlm("data/smallmol/hpopt_rsm_$simid.text", out)
        if save_folds
            save("data/smallmol/folds_$simid.jld", "data", foldss)
        end
    end
end


"""
!! for earlier ver, the hyperparameters are in index 5:8, current ver: 5:7 (true/false removed, shift -1 index)
batch run using params obtained from hpopt
e.g.:
 params = readdlm("data/smallmol/hpopt_hxoy_rsm.text", '\t')
 data = load("data/smallmol/hxoy_data_req.jld", "data")
 sim_id = replace(replace(string(now()), "-"=>""), ":"=>"")[1:end-4]
 main_eval_rsm(data, params; sim_id = sim_id)

example script in vsc:
 main_eval_rsm(load("data/smallmol/hxoy_data_req.jld", "data")[setdiff(1:15, 10)], readdlm("data/smallmol/hpopt_rsm_hxoy_20241107T124925.text", '\t'); sim_id=strnow, foldss=load("data/smallmol/folds_hxoy_20241107T124925.jld", "data"))
"""
function main_eval_rsm(data, hpopt_params; sim_id="", foldss=[])
    Random.seed!(603) # still set seed for folds
    # match each result with the corresponding dataset
    λ = 0.
    ld_res = []
    for d ∈ data
        # determine which result id:
        id = findall(d["mol"] .== hpopt_params[:,1])[1]
        hp = hpopt_params[id,:]
        println([d["mol"], hp[5:end]])
        # load data:
        E = d["V"]
        F = rdist.(d["R"], d["req"], c=hp[5]);
        folds = [] # reinitialization
        if isempty(foldss)
            folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        else
            for cfold in foldss
                if cfold["mol"] == d["mol"]
                    folds = cfold["folds"]
                end
            end
        end
        display(folds)
        println(hp[5:7])
        MAEs, RMSEs, RMSDs, t_lss, t_preds = rosemi_fitter(F, E, folds; n_basis=hp[6], ptr=hp[7], force=false, λ = λ)
        println(mean(RMSEs))
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"];
        push!(ld_res, d_res)
    end
    save("result/hdrsm_$sim_id.jld", "data", ld_res)
end

"""
singlet eval only

example script in VSC:
 main_single_eval_rsm(load("data/smallmol/hxoy_data_req.jld", "data")[10], readdlm("data/smallmol/hpopt_rsm_hxoy_20241107T124925.text", '\t')[9,:]; sim_id="H2_2_0711")
"""
function main_single_eval_rsm(data, hp; sim_id="", folds=[], λ = 0.)
    E = data["V"]
    F = rdist.(data["R"], data["req"], c=hp[5])
    if isempty(folds)
        folds = shuffle.(collect(Kfold(length(E), 5)))
    end
    MAEs, RMSEs, RMSDs, t_lss, t_preds = rosemi_fitter(F, E, folds; n_basis=hp[6], ptr=hp[7], force=false, λ = λ)
    println(mean(RMSEs))
    d_res = Dict()
    d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
    d_res["mol"] = data["mol"];
    display(d_res)
    save("result/hdrsm_singlet_$sim_id.jld", "data", d_res)
end