using Krylov, Flux, Printf, DelimitedFiles
using ProgressMeter, Dates, BenchmarkTools
using Random, StatsBase

"""
contains all tests and experiments
"""

include("voronoi.jl")
include("linastic.jl")
include("RoSemi.jl")

"""
timer callback for earlier stop by time duration
"""
function time_callback(solver::CglsSolver, start_time, duration)
    return time()-start_time ≥ duration
end

function get_index(molname, D)
    indices = []
    @simd for i ∈ eachindex(D)
        if D[i]["formula"] == molname
            push!(indices, i)
        end
    end
    return indices
end

"""
get the index of data that has n_data >= 200
"""
function find_data()
    D = load("data/qm9_dataset.jld")["data"]
    formulas= collect(Set([d["formula"] for d in D]))
    indices = map(formula -> get_index(formula, D), formulas) # a vector of vector
    # transform to dictionary:
    D = Dict()
    for i ∈ eachindex(indices) # eachindex returns an iterator, which means avoids materialization/alloc → more efficient, should use more iterators wherever possible
        D[formulas[i]] = convert(Vector{Int}, indices[i])
    end
    save("data/formula_indices.jld", "data", D)
end

"""
filter data which has num of data >= n_data
"""
function filter_indices(n_data)
    ind = load("data/formula_indices.jld")["data"]
    filtered = Dict()
    for k ∈ keys(ind)
        if length(ind[k]) >= n_data
            filtered[k] = ind[k]
        end
    end
    return filtered
end

"""
get the indices of the supervised datapoints M, fix w0 as i=603 for now
params:
    - M, number of supervised data points
    - universe_size, the total available data, default =1000, although for complete data it should be 130k
"""
function set_cluster(infile, M; universe_size=1_000)
    # load dataset || the datastructure is subject to change!! since the data size is huge
    dataset = load(infile)["data"]
    N, L = size(dataset)
    A = dataset' # transpose data (becomes column major)
    println(N, " ", L)
    # compute mean and cov:
    idx = 603 # the ith data point of the dataset, can be arbitrary technically, for now fix 603:= RoZeMi 🌹
    idx = Int(round(idx/universe_size*N)) # relative idx
    wbar, C = mean_cov(A, idx, N, L)
    B = Matrix{Float64}(I, L, L) # try with B = I #B = compute_B(C) 
    # generate centers (M) for training:
    center_ids, mean_point, D = eldar_cluster(A, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd", get_distances=true) # generate cluster centers
    display(center_ids)
    return center_ids, D
end

"""
overloader, using matrix as input
"""
function set_cluster(F::Matrix{Float64}, M::Int; universe_size=1_000, num_center_sets::Int = 1)
    N, L = size(F)
    Ftran = Matrix(transpose(F))
    #F = F'
    #idx = 603 # the ith data point of the dataset, can be arbitrary technically, for now fix 603:= RoZeMi 🌹
    #idx = Int(round(idx/universe_size*N)) # relative idx
    #wbar, C = mean_cov(F, idx, N, L)
    #B = Matrix{Float64}(I, L, L) # try with B = I #B = compute_B(C) 
    # generate centers (M) for training:
    #center_ids, mean_point, D = eldar_cluster(F, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd", get_distances=true) # generate cluster centers
    centers = []
    for i ∈ 1:num_center_sets
        _, center_ids = usequence(Ftran, M)
        push!(centers, center_ids)
    end
    return centers
end

"""
compute all D_k(w_l) ∀k,l, for now fix i = 603 (rozemi)
"""
function set_all_dist(infile; universe_size=1_000)
    dataset = load(infile)["data"]
    N, L = size(dataset)
    W = dataset' # transpose data (becomes column major)
    println(N, " ", L)
    # compute mean and cov:
    idx = 603 # the ith data point of the dataset, can be arbitrary technically, for now fix 603:= RoZeMi 🌹
    idx = Int(round(idx/universe_size*N))
    wbar, C = mean_cov(W, idx, N, L)
    B = Matrix{Float64}(I, L, L) # try with B = I #B = compute_B(C) 
    #display(wbar)
    #dist = f_distance(B, A[:,1], A[:,2])
    #display(dist)

    # compute all distances:
    D = compute_distance_all(W, B)
    return D, idx
end

"""
setup all the data files needed for fitting a certain molecule
"""
function data_setup(mol_name, n_data, n_feature, M; universe_size=1_000)
    println("data setup for mol=",mol_name,", n_data=", n_data,", n_feature=",n_feature,", M=", M, " starts!")
    t = @elapsed begin
        # create subfolder:
        path = mkpath("data/$mol_name")
        # query (get the index of data) the molecule by molecule name:
        D = load("data/qm9_dataset.jld")["data"] # 🌸
        len = length(D)
        indexes = []
        for i ∈ 1:len
            if D[i]["formula"] == mol_name
                push!(indexes, i)
            end
        end
        # cut data into n_data:
        l = length(indexes)
        if l >= n_data
            indexes = indexes[1:n_data]
        end
        D = D[indexes]
        println(length(D))
        save(path*"/$mol_name"*"_dataset_$n_data.jld", "data", D)
        # try PCA -> slice -> normalize:
        #= W = load("data/ACSF_PCA.jld")["data"]   # PCA
        W = W[indexes, 1:n_feature]     # slice
        W = normalize_routine(W)        # normalize =#
        # slice the global feature matrix:
        W = load("data/ACSF_PCA_scaled.jld")["data"] # load scaled(PCA(features)), this is more accurate since the columns are sorted by the most important featuers
        W = W[indexes, 1:n_feature] # slice the featuere matrix by the data indices and the first n_feature
        # try using binomial features:
        #= W = load("data/ACSF_PCA_bin_$n_feature"*"_scaled.jld")["data"] # need to change the flow later since the PCA and scaling is fast to compuite
        W = W[indexes, :] =#
        main_file = path*"/$mol_name"*"_ACSF_"*"$n_feature"*"_"*"$n_data.jld"
        save(main_file, "data", W)
        # get center indexes:
        M_actual = M
        if M > n_data # check if the wanted centers is too much for the data..
            M_actual = n_data
        end
        center_ids, Dist = set_cluster(main_file, M_actual, universe_size=universe_size)
        save(path*"/$mol_name"*"_M=$M"*"_$n_feature"*"_$n_data.jld", "data", center_ids)
        # compute all distances:
        #Dist, idx = set_all_dist(main_file, universe_size=universe_size)
        save(path*"/$mol_name"*"_distances_"*"$n_feature"*"_$n_data.jld", "data", Dist)
        # scale feature for basis:
        #= W = normalize_routine(main_file)
        save(path*"/$mol_name"*"_ACSF_"*"$n_feature"*"_"*"$n_data"*"_symm_scaled.jld", "data", W) =#
    end
    println("data setup for mol=",mol_name,", n_data=", n_data,", n_feature=",n_feature,", M=", M, " is finished in $t seconds!!")
end

"""
overloader for data setup, manually input filedir of the feature here
"""
function data_setup(mol_name, n_data, n_feature, M, feature_file; universe_size=1_000)
    println("data setup for mol=",mol_name,", n_data=", n_data,", n_feature=",n_feature,", M=", M, " starts!")
    t = @elapsed begin
        # create subfolder:
        path = mkpath("data/$mol_name")
        # query (get the index of data) the molecule by molecule name:
        D = load("data/qm9_dataset.jld")["data"] # 🌸
        len = length(D)
        indexes = []
        for i ∈ 1:len
            if D[i]["formula"] == mol_name
                push!(indexes, i)
            end
        end
        # cut data into n_data:
        l = length(indexes)
        if l >= n_data
            indexes = indexes[1:n_data]
        end
        D = D[indexes]
        println(length(D))
        save(path*"/$mol_name"*"_dataset_$n_data.jld", "data", D)

        W = load(feature_file)["data"] # load scaled(PCA(features)), this is more accurate since the columns are sorted by the most important featuers
        W = W[indexes, :] # slice the featuere matrix by the data indices and the first n_feature
        
        main_file = path*"/$mol_name"*"_ACSF_"*"$n_feature"*"_"*"$n_data.jld"
        save(main_file, "data", W)
        # get center indexes:
        M_actual = M
        if M > n_data # check if the wanted centers is too many for the data..
            M_actual = n_data
        end
        center_ids, Dist = set_cluster(main_file, M_actual, universe_size=universe_size)
        save(path*"/$mol_name"*"_M=$M"*"_$n_feature"*"_$n_data.jld", "data", center_ids)
        # compute all distances:
        save(path*"/$mol_name"*"_distances_"*"$n_feature"*"_$n_data.jld", "data", Dist)

    end
    println("data setup for mol=",mol_name,", n_data=", n_data,", n_feature=",n_feature,", M=", M, " is finished in $t seconds!!")
end

"""
full data setup, in contrast to each molecule data setup, INCLUDES PCA!.
takes in the data indices (relative to the qm9 dataset).
if molf_file  is not empty then there will be no atomic feature extractions, only PCA on molecular level
"""
function data_setup(foldername, n_af, n_mf, n_basis, num_centers, dataset_file, feature_file, feature_name; 
                    universe_size=1_000, normalize_atom = true, normalize_mol = true, normalize_mode = "minmax", 
                    fit_ecdf = false, fit_ecdf_ids = [], ft_sos=false, ft_bin=false, 
                    molf_file = "", cov_file = "", sensitivity_file = "", save_global_centers = false, num_center_sets = 1)
    println("data setup for atom features = ",n_af, ", mol features = ", n_mf, ", centers = ",num_centers, " starts!")
    t = @elapsed begin
        path = mkpath("data/$foldername")
        # load dataset:
        dataset = load(dataset_file)["data"]
        # PCA:
        F = nothing
        sens_mode = false
        uid = replace(string(Dates.now()), ":" => ".") # generate uid
        plot_fname = "$foldername"*"_$uid"*"_$feature_name"*"_$n_af"*"_$n_mf"*"_$ft_sos"*"_$ft_bin" # plot name infix
        if length(molf_file) == 0 # if molecular feature file is not provided:
            println("atomic ⟹ mol mode!")
            f = load(feature_file)["data"] # pre-extracted atomic features
            println("PCA atom starts!")
            if isempty(cov_file)
                f = PCA_atom(f, n_af; fname_plot_at=plot_fname, normalize=normalize_atom)
            else
                sens_mode = true
                C = load(cov_file)["data"]
                σ = load(sensitivity_file)["data"]
                f = PCA_atom(f, n_af, C, σ; fname_plot_at=plot_fname, normalize=normalize_atom)
            end
            println("PCA atom done!")
            println("mol feature processing starts!")
            F = extract_mol_features(f, dataset; ft_sos = ft_sos, ft_bin = ft_bin)
            F = PCA_mol(F, n_mf; fname_plot_mol=plot_fname, normalize=normalize_mol, normalize_mode=normalize_mode)
            println("mol feature processing finished!")
        else
            println("mol only mode!")
            println("mol feature processing starts!")
            F = load(molf_file)["data"]
            F = PCA_mol(F, n_mf, fname_plot_mol = plot_fname, normalize=normalize_mol, normalize_mode=normalize_mode, cov_test=feature_name=="FCHL" ? true : false)
            println("mol feature processing finished!")
        end
        #F = F[data_indices, :]; f = f[data_indices]
        #dataset = dataset[data_indices] # slice dataset
        # compute bspline:
        ϕ, dϕ = extract_bspline_df(F', n_basis; flatten=true, sparsemat=true) # move this to data setup later
        # get centers:
        println("computing centers...")
        centers = set_cluster(F, num_centers, universe_size=universe_size, num_center_sets=num_center_sets)
        if fit_ecdf # normalization by fitting the ecdf using the centers
            println("fitting ecdf...")
            ids = []
            if isempty(fit_ecdf_ids) # if empty then use the first index centers
                ids = centers[1]
            else
                ids = fit_ecdf_ids
            end
            f = comp_ecdf(f, ids; type="atom")
            F = comp_ecdf(F, ids; type="mol")
        end
        # copy pre-computed atomref features:
        redf = load("data/atomref_features.jld", "data")
        # save files:
    end
    #save("data/$foldername/dataset.jld", "data", dataset)
    save("data/$foldername/features_atom.jld", "data", f) # atomic features
    save("data/$foldername/features.jld", "data", F) # molecular features
    save("data/$foldername/atomref_features.jld", "data", redf) # features to compute sum of atomic energies
    save("data/$foldername/center_ids.jld", "data", centers)
    save("data/$foldername/spline.jld", "data", ϕ)
    save("data/$foldername/dspline.jld", "data", dϕ)
    if save_global_centers # append centers to global directory
        for i ∈ eachindex(centers)
            kid = "K"*string(i)
            strings = string.(vcat(uid, kid, centers[i]))
            open("data/centers.txt", "a") do io
                str = ""
                for s ∈ strings
                    str*=s*"\t"
                end
                print(io, str*"\n")
            end
        end
    end
    # write data setup info:
    n_data = length(dataset)
    machine = splitdir(homedir())[end]; machine = machine=="beryl" ? "SAINT" : "OMP1" # machine name
    strlist = string.([uid, n_data, num_centers, feature_name, n_af, n_mf, ft_sos, ft_bin, normalize_atom, normalize_mol, sens_mode, n_basis+3, t, machine]) # dates serves as readable uid, n_basis + 3 by definition
    open("data/$foldername/setup_info.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    println("data setup is finished in ",t,"s")
    # clear memory:
    dataset=F=f=ϕ=dϕ=centers=redf=nothing
    GC.gc()
end

"""
data setup which only include atomic features ⟹ no data selection
"""
function data_setup_atom(foldername, n_af, dataset_file, feature_file, feature_name; 
                    universe_size=1_000, normalize_atom = true, normalize_mode = "minmax", 
                    fit_ecdf = false, fit_ecdf_ids = [], 
                    cov_file = "", sensitivity_file = "")
    println("data setup for ATOMIC FEATURES only = ",n_af," starts!")
    t = @elapsed begin
        path = mkpath("data/$foldername")
        # load dataset:
        dataset = load(dataset_file)["data"]
        # PCA:
        F = nothing
        sens_mode = false
        uid = replace(string(Dates.now()), ":" => ".") # generate uid
        plot_fname = "$foldername"*"_$uid"*"_$feature_name"*"_$n_af" # plot name infix
        
        println("atomic ⟹ mol mode!")
        f = load(feature_file)["data"] # pre-extracted atomic features
        println("PCA atom starts!")
        if isempty(cov_file)
            f = PCA_atom(f, n_af; fname_plot_at=plot_fname, normalize=normalize_atom)
        else
            sens_mode = true
            C = load(cov_file)["data"]
            σ = load(sensitivity_file)["data"]
            f = PCA_atom(f, n_af, C, σ; fname_plot_at=plot_fname, normalize=normalize_atom)
        end
        println("PCA atom done!")
        ϕ = dϕ = F = [] # not needed for atomic level models
        if fit_ecdf # normalization by fitting the ecdf using the centers
            println("fitting ecdf...")
            ids = fit_ecdf_ids
            f = comp_ecdf(f, ids; type="atom")
        end
        # copy pre-computed atomref features:
        redf = load("data/atomref_features.jld", "data")
        # save files:
    end
    #save("data/$foldername/dataset.jld", "data", dataset)
    save("data/$foldername/features_atom.jld", "data", f) # atomic features
    save("data/$foldername/features.jld", "data", F) # molecular features
    save("data/$foldername/atomref_features.jld", "data", redf) # features to compute sum of atomic energies
    save("data/$foldername/spline.jld", "data", ϕ)
    save("data/$foldername/dspline.jld", "data", dϕ)

    # write data setup info:
    n_data = length(dataset)
    machine = splitdir(homedir())[end]; machine = machine=="beryl" ? "SAINT" : "OMP1" # machine name
    strlist = string.([uid, n_data, feature_name, n_af, normalize_atom, sens_mode, t, machine]) # dates serves as readable uid, n_basis + 3 by definition
    open("data/$foldername/setup_info.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    println("data setup is finished in ",t,"s")
    # clear memory:
    dataset=F=f=ϕ=dϕ=redf=nothing
    GC.gc()
end

"""
this takes a very long time to finish, jsutified as a part of data setup
"""
function compute_sigma2(foldername)
    F = load("data/$foldername/features.jld")["data"]
    display(F)
    t = @elapsed begin
        σ2 = get_sigma2(F)    
    end
    save("data/$foldername/sigma2.jld", "data", σ2)
    println("sigma computation time = ", t)
end

"""
this only compute the number of atoms fo each type as feature, and do fmd on it to get the centers
"""
function mini_data_setup(foldername, file_dataset, data_indices, n_basis, num_centers; universe_size=1_000)
    t = @elapsed begin
        path = mkpath("data/$foldername")
        dataset = load(file_dataset)["data"]   # load dataset
        F = get_atom_counts(dataset) #  get the atomcounts
        dataset = dataset[data_indices] # slice
        F = F[data_indices, :]
        # compute bspline:
        ϕ, dϕ = extract_bspline_df(F', n_basis; flatten=true, sparsemat=true) # move this to data setup later
        # get centers:
        center_ids, distances = set_cluster(F, num_centers, universe_size=universe_size)
        display(center_ids)
    end
    save("data/$foldername/dataset.jld", "data", dataset)
    save("data/$foldername/features.jld", "data", F) # molecular features
    save("data/$foldername/center_ids.jld", "data", center_ids)
    save("data/$foldername/distances.jld", "data", distances)
    save("data/$foldername/spline.jld", "data", ϕ)
    save("data/$foldername/dspline.jld", "data", dϕ)
    # write data setup info:
    n_data = length(data_indices)
    strlist = string.([n_data, n_basis+3, num_centers]) # n_basis + 3 by definition
    open("data/$foldername/setup_info.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    println("data setup is finished in ",t,"s")
end


"""
main fitter function, assemble LS -> fit -> save to file
to avoid clutter in main function, called within fitting iters

outputs:
    - indexes of n-maximum MAD
"""
function fitter(F, E, D, ϕ, dϕ, Midx, Tidx, Uidx, Widx, n_feature, mol_name, bsize, tlimit; get_mad=false, Er=Vector{Float64}()::Vector{Float64})
    N = length(Tidx); nU = length(Uidx); nK = length(Midx); Nqm9 = length(Widx)
    nL = size(ϕ, 1); n_basis = nL/n_feature
    println("[Nqm9, N, nK, nf, ns, nL] = ", [Nqm9, N, nK, n_feature, n_basis, nL])   

    # !!!! using LinearOperators !!!:
    # precompute stuffs:
    t_ab = @elapsed begin
        # indexers:
        klidx = kl_indexer(nK, nL)
        cidx = 1:nK
        # intermediate value:
        SKs_train = map(m -> comp_SK(D, Midx, m), Uidx) # only for training, disjoint index from pred
        γ = comp_γ(D, SKs_train, Midx, Uidx)
        SKs = map(m -> comp_SK(D, Midx, m), Widx) # for prediction
        α = γ .- 1
        B = zeros(nU, nK*nL); comp_B!(B, ϕ, dϕ, F, Midx, Uidx, nL, n_feature);
    end
    println("precomputation time = ",t_ab)
    row = nU*nK; col = nK*nL #define LinearOperator's size
    t_ls = @elapsed begin
        # generate LinOp in place of A!:
        Axtemp = zeros(nU, nK); tempsA = [zeros(nU) for _ in 1:3]
        op = LinearOperator(Float64, row, col, false, false, (y,u) -> comp_Ax!(y, Axtemp, tempsA, u, B, Midx, cidx, klidx, γ, α), 
                                                            (y,v) -> comp_Aᵀv!(y, v, B, Midx, Uidx, γ, α, nL))
        show(op)
        # generate b:
        b = zeros(nU*nK); btemp = zeros(nU, nK); tempsb = [zeros(nU) for _ in 1:2]
        comp_b!(b, btemp, tempsb, E, γ, α, Midx, cidx)
        # do LS:
        start = time()
        θ, stat = cgls(op, b, itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit)) # with callback 🌸
        #θ, stat = cgls(op, b, itmax=500, verbose=0) # without ccallback
    end

    # get residual:
    obj = norm(op*θ - b)^2
    println("solver obj = ",obj, ", solver time = ",t_ls)

    # get residuals of training set:
    VK = zeros(nU); outs = [zeros(nU) for _ = 1:3]
    comp_VK!(VK, outs, E, D, θ, B, SKs_train, Midx, Uidx, cidx, klidx)
    v = zeros(nU*nK); vmat = zeros(nU, nK); fill!.(outs, 0.)
    comp_res!(v, vmat, outs, VK, E, θ, B, klidx, Midx, α)
    MADs = vec(sum(abs.(vmat), dims=2)) ./ nK # length nU

    # semi-BATCHMODE PRED for Nqm9:
    blength = Nqm9 ÷ bsize # number of batch iterations
    batches = kl_indexer(blength, bsize)
    bend = batches[end][end]
    bendsize = Nqm9 - (blength*bsize)
    push!(batches, bend+1 : bend + bendsize)
    # compute predictions:
    t_batch = @elapsed begin
        VK_fin = zeros(Nqm9)
        B = zeros(Float64, bsize, nK*nL)
        VK = zeros(bsize); outs = [zeros(bsize) for _ = 1:3]
        @simd for batch in batches[1:end-1]
            comp_B!(B, ϕ, dϕ, F, Midx, Widx[batch], nL, n_feature)
            comp_VK!(VK, outs, E, D, θ, B, SKs[batch], Midx, Widx[batch], cidx, klidx)
            VK_fin[batch] .= VK
            # reset:
            fill!(B, 0.); fill!(VK, 0.); fill!.(outs, 0.); 
        end
        # remainder part:
        B = zeros(Float64, bendsize, nK*nL)
        VK = zeros(bendsize); outs = [zeros(bendsize) for _ = 1:3]
        comp_B!(B, ϕ, dϕ, F, Midx, Widx[batches[end]], nL, n_feature)
        comp_VK!(VK, outs, E, D, θ, B, SKs[batches[end]], Midx, Widx[batches[end]], cidx, klidx)
        VK_fin[batches[end]] .= VK
        VK = VK_fin # swap
    end
    if !isempty(Er) # check if the energy for fitting were reduced
       VK .+= Er[Widx] 
    end
    println("batchpred time = ",t_batch)

    # get errors: 
    MAE = sum(abs.(VK .- E[Widx])) / Nqm9
    MAE *= 627.503 # convert from Hartree to kcal/mol
    println("MAE of all mol w/ unknown E is ", MAE)

    # get the n-highest MAD:
    n = 1 # 🌸
    sidxes = sortperm(MADs)[end-(n-1):end]
    MADmax_idxes = Widx[sidxes] # the indexes relative to Widx (global data index)
    
    # get min |K| RMSD (the obj func):
    RMSD = obj #Optim.minimum(res)
    
    println("largest MAD is = ", MADs[sidxes[end]], ", with index = ",MADmax_idxes)
    println("|K|*∑RMSD(w) = ", RMSD)

    # save all errors foreach iters:
    data = [MAE, RMSD, MADs[sidxes[end]]]
    matsize = [Nqm9, nK, nU, n_feature, n_basis]
    #strlist = string.(vcat(MAE, matsize, [t_ab, t_ls, t_batch]))
    strlist = string.([MAE, Nqm9, nK, n_feature, t_ls, t_batch])
    open("result/$mol_name/err_$mol_name.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    # save also the nK indices and θ's to file!!:
    data = Dict("centers"=>Midx, "theta"=>θ)
    save("result/$mol_name/theta_center_$mol_name"*"_$matsize.jld", "data", data)
    # clear variables:
    SKs_train = SKs = γ = α = B = klidx = cidx = Axtemp = tempsA = op = b = tempsb = θ = stat = VK = outs = v = vmat = MADs = batches = VK_fin = nothing; GC.gc()
    return MAE, MADmax_idxes
end

"""
the main fitting function !!!
targets: chemical accuracy = 1 kcal/mol = 0.0015936 Ha = 0.0433641 eV.
try:
    - use fixed N while varying M (recomputing centers)
    - changing column length
    - multirestart
"""
function fit_🌹(mol_name, n_data, n_feature, M)
    println("FITTING MOL: $mol_name")
    # required files:
    #= files = readdir("data/$mol_name"; join=true)
    file_finger, file_centers, file_dataset, file_distance = files[2:end] =#
    path = "data/$mol_name/"
    file_dataset = path*"$mol_name"*"_dataset_$n_data.jld"
    file_finger = path*"$mol_name"*"_ACSF_$n_feature"*"_$n_data.jld"
    file_distance = path*"$mol_name"*"_distances_$n_feature"*"_$n_data.jld"
    file_centers = path*"$mol_name"*"_M=$M"*"_$n_feature"*"_$n_data.jld"
    # result files:
    mkpath("result/$mol_name")

    # setup parameters:
    n_basis = 5 # pre-inputted number, later n_basis := n_basis+3 🌸
    dataset = load(file_dataset)["data"] # energy is from here
    W = load(file_finger)["data"]' # load and transpose the normalized fingerprint (sometime later needs to be in feature × data format already so no transpose)
    s_W = size(W) # n_feature × n_data
    n_feature = s_W[1]; n_data = s_W[2];
    E = map(d -> d["energy"], dataset)
    #println(E)
    D = load(file_distance)["data"] # the mahalanobis distance matrix
    # index op:
    data_idx = 1:n_data
    Midx_g = load(file_centers)["data"] # the global supervised data points' indices
    
    ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true) # compute basis from fingerprint ∈ (n_feature*(n_basis+3), n_data)
    n_basis += 3 # by definition of bspline
    #display(ϕ)
    #display([nnz(ϕ), nnz(dϕ)]) # only ≈1/3 of total entry is nnz
    #display(Base.summarysize(ϕ)) # turns out only 6.5mb for sparse
    println("[feature, basis]",[n_feature, n_basis])
    # === compute!! ===:
    inc_M = 10 # 🌸
    MADmax_idxes = nothing; Midx = nothing; Widx = nothing # set empty vars
    thresh = 0.9 # .9 kcal/mol desired acc 🌸
    for i ∈ [10] # M iter increment
        Midx = Midx_g[1:inc_M*i] # the supervised data
        Widx = setdiff(data_idx, Midx) # the unsupervised data, which is ∀i w_i ∈ W \ K, "test" data
        #Widx = Widx[1:30] # take subset for smaller matrix
        println("======= LOOP i=$i =======")
        MAE, MADmax_idxes = fitter(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis, mol_name)
        if MAE < thresh # in kcal/mol
            println("desirable MAE reached!!")
            break
        end
        println()
    end


    # N iter increment mode, start from 150


    #= # use the info of MAD for fitting :
    for i ∈ 1:10
        #println(i,", max MAD indexes from the last loop = ", MADmax_idxes)
        Midx = vcat(Midx, MADmax_idxes) # put the n-worst MAD as centers
        filter!(e->e ∉ MADmax_idxes, Widx) # cut the n-worst MAD from unsupervised data
        println("======= MAD mode, LOOP i=$i =======")
        MAE, MADmax_idxes = fitter(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis, mol_name, get_mad = true)
        if MAE < thresh # in kcal/mol
            println("desirable MAE reached!!")
            break
        end
        println()
    end
    println() =#
end

"""
fit overloader, for custom data indices, 
and new indexing mode: fitted with data from w ∈ T ∉ K (unsupervised), where centers = T, hence the centers should be larger than previous one now (>100)
"""
function fit_🌹(foldername, bsize, tlimit; mad = false, reduced_E = false)
    println("FITTING: $foldername")
    # input files:
    path = "data/$foldername/"
    mkpath("result/$foldername")
    file_dataset = path*"dataset.jld"
    file_finger = path*"features.jld"
    file_distance = path*"distances.jld"
    file_centers = path*"center_ids.jld"
    file_spline = path*"spline.jld"
    file_dspline = path*"dspline.jld"
    files = [file_dataset, file_finger, file_distance, file_centers, file_spline, file_dspline]
    dataset, F, D, Tidx, ϕ, dϕ = [load(f)["data"] for f in files]
    
    F = F' # always transpose
    E = map(d -> d["energy"], dataset)

    # set empty vars:
    MADmax_idxes = nothing; Midx = nothing; Widx = nothing 
    thresh = 0.9 # .9 kcal/mol desired acc 🌸
    if mad
        inc_M = 10 # 🌸
        for i ∈ [9] # M iter increment 🌸
            Midx = Midx_g[1:inc_M*i] # the supervised data
            Widx = setdiff(data_idx, Midx) # the unsupervised data, which is ∀i w_i ∈ W \ K, "test" data
            #Widx = Widx[1:30] # take subset for smaller matrix
            println("======= LOOP i=$i =======")
            MAE, MADmax_idxes = fitter(F, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis, foldername)
            if MAE < thresh # in kcal/mol
                println("desirable MAE reached!!")
                break
            end
            println()
        end
        for i ∈ 1:10 #🌸
            #println(i,", max MAD indexes from the last loop = ", MADmax_idxes)
            Midx = vcat(Midx, MADmax_idxes) # put the n-worst MAD as centers
            filter!(e->e ∉ MADmax_idxes, Widx) # cut the n-worst MAD from unsupervised data
            println("======= MAD mode, LOOP i=$i =======")
            MAE, MADmax_idxes = fitter(F, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis, foldername, get_mad = true)
            if MAE < thresh # in kcal/mol
                println("desirable MAE reached!!")
                break
            end
            println()
        end
    else
        # compute indices:
        n_data = length(dataset); n_feature = size(F, 1);
        Midx = Tidx[1:100] # 🌸 for now
        Uidx = setdiff(Tidx, Midx) # (U)nsupervised data
        Widx = setdiff(1:n_data, Midx) # for evaluation 
        if reduced_E # if we use the reduced energy, then subtract the energy for training
            Er = load("data/energy_reducer.jld", "data")
            E[Midx] .-= Er["L1"][Midx] # we take L1 recipe for now 
        end
        MAE, MADmax_idxes = fitter(F, E, D, ϕ, dϕ, Midx, Tidx, Uidx, Widx, n_feature, foldername, bsize, tlimit; Er = Er["L1"]) #
    end
end


"""
naive linear least squares, take whatever feature extracted
"""
function fitter_LLS(F, E, Midx, Widx, foldername, tlimit; Er = Vector{Float64}()::Vector{Float64})
    nK = length(Midx); Nqm9 = length(Widx); n_f = size(F, 2)
    A = F[Midx, :] # construct the data matrix
    start = time()
    t_ls = @elapsed begin
        θ, stat = cgls(A, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    end
    # check MAE of training data only:
    errors = abs.(A*θ - E[Midx]) .* 627.503 # in kcal/mol
    MAEtrain = sum(errors)/length(errors)
    println("MAE train = ",MAEtrain)
    t = @elapsed begin
        E_pred = F[Widx, :]*θ # pred, should be fast
    end
    if !isempty(Er) # check if the energy for fitting were reduced
        E_pred .+= Er[Widx] 
    end
    errors = abs.(E_pred - E[Widx]) .* 627.503 # in kcal/mol
    MAE = sum(errors)/length(errors)
    println("MAE of Nqm9 = ",MAE)
    # save info to file
    strlist = string.([MAE, Nqm9, nK, n_f, t_ls, t])
    open("result/$foldername/err_$foldername.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    # clear memory:
    A=θ=stat=errors=E_pred=nothing; GC.gc()
end

function fit_LLS(foldername, bsize, tlimit; reduced_E = false, train_labels = Vector{Int64}()::Vector{Int64})
    println("FITTING LLS: $foldername")
    # file loaders:
    path = "data/$foldername/"
    mkpath("result/$foldername")
    file_dataset = path*"dataset.jld"
    file_finger = path*"features.jld" 
    file_centers = path*"center_ids.jld"
    files = [file_dataset, file_finger, file_centers]
    dataset, F, Tidx = [load(f)["data"] for f in files]
    E = map(d -> d["energy"], dataset)
    # compute indices:
    n_data = length(dataset);
    K_indexer = 1:100 # 🌸 temporary selection
    Midx = Tidx[K_indexer] 
    if !isempty(train_labels)
        Midx = train_labels
    end
    Widx = setdiff(1:n_data, Midx) # for evaluation 
    A = F[Midx, :] # construct the data matrix
    start = time()
    θ, stat = cgls(A, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    # check MAE of training data only:
    errors = abs.(A*θ - E[Midx]) .* 627.503 # in kcal/mol
    MAEtrain = sum(errors)/length(errors)
    println("MAE train = ",MAEtrain)

    E_pred = F[Widx, :]*θ # pred, should be fast
    errors = abs.(E_pred - E[Widx]) .* 627.503 # in kcal/mol
    MAE = sum(errors)/length(errors)
    println("MAE of Nqm9 = ",MAE)
    return F*θ # this computes the sum of atomic energies, used for the reduced energy later # remove this later 
end

"""
gaussian kernel mode
"""
function fitter_KRR(F, E, Midx, Tidx, Widx, K_indexer, foldername, tlimit, n_feature; scaler = 2.0 * (2.0 ^ 5) ^ 2, Er = Vector{Float64}()::Vector{Float64})
    nK = length(Midx); Nqm9 = length(Widx)
    t_pre = @elapsed begin
        Norms = get_norms(F, Tidx, Midx)
        #σ0 =  get_sigma0(Norms)
        #scaler = 1. # 🌸 hyperparameter   
        σ2 = scaler #σ2 = scaler * σ0
        comp_gaussian_kernel!(Norms, σ2) # generate the kernel
        K = Norms[K_indexer, K_indexer] # since the norm matrix' entries are changed
    end
    display(K)
    println("pre-computation time is ",t_pre)
    # do LS:
    start = time()
    t_ls = @elapsed begin
        θ, stat = cgls(K, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    end
    display(stat)
    # check MAE of training data only:
    errors = abs.(K*θ - E[Midx]) .* 627.503
    MAEtrain = sum(errors)/length(errors)
    # prediction (MOL GAUSS MODE!!):
    t_pred = @elapsed begin
        K_pred = get_norms(F, Widx, Midx)
        comp_gaussian_kernel!(K_pred, σ2)
        E_pred = K_pred*θ
    end
    if !isempty(Er) # check if the energy for fitting were reduced
        E_pred .+= Er[Widx] 
    end
    errors = abs.(E_pred - E[Widx]) .* 627.503
    MAE = sum(errors)/length(errors)
    #println([σ0, σ2])
    println("pre-computation time is ",t_pre, ", MAEtrain=",MAEtrain)
    println("MAE of Nqm9 = ",MAE, ", with t_pred = ", t_pred)
    # save info to file
    strlist = string.([MAE, Nqm9, nK, n_feature, t_ls, t_pred])
    open("result/$foldername/err_$foldername.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    # clear variable:
    Norms=K=θ=stat=errors=K_pred=E_pred=Er=nothing; GC.gc()
end

function fit_KRR(foldername, bsize, tlimit; reduced_E = false, train_labels = Vector{Int64}()::Vector{Int64})
    println("FITTING KRR: $foldername")
    # input files:
    path = "data/$foldername/"
    mkpath("result/$foldername")
    file_dataset = path*"dataset.jld"
    file_finger = path*"features.jld" # file_finger = path*"features_atom.jld" #
    #file_distance = path*"distances.jld"
    file_centers = path*"center_ids.jld"
    #file_σ2 = path*"sigma2.jld"
    #file_spline = path*"spline.jld"
    #file_dspline = path*"dspline.jld"
    files = [file_dataset, file_finger, file_centers]
    dataset, F, Tidx = [load(f)["data"] for f in files]
    E = map(d -> d["energy"], dataset)
    # compute indices:
    n_data = length(dataset);
    K_indexer = 1:100 # 🌸 temporary selection
    Midx = Tidx[K_indexer] 
    Uidx = setdiff(Tidx, Midx) # (U)nsupervised data
    Widx = setdiff(1:n_data, Midx) # for evaluation 
    # compute hyperparams (MOLECULAR GAUSSIAN MODE!): # ⭐
    t_pre = @elapsed begin
        Norms = get_norms(F, Tidx, Midx)
        σ0 =  get_sigma0(Norms)
        scaler = 1. # 🌸 hyperparameter   
        σ2 = scaler * σ0
        comp_gaussian_kernel!(Norms, σ2) # generate the kernel
        K = Norms[K_indexer, K_indexer] # since the norm matrix' entries are changed
    end
    # ATOMIC GAUSSIAN MODE: # ⭐
   #=  t_pre = @elapsed begin
        Norms = get_norms_at(F, Tidx, Midx)
        σ0 =  get_sigma0_at(Norms)
        scaler = 1. # 🌸 hyperparameter   
        σ2 = scaler * σ0
        K = comp_gaussian_kernel_at(Norms, σ2) # generate the kernel
        K = K[K_indexer, K_indexer] # since the norm matrix' entries are changed
    end =#
    display(K)
    println("pre-computation time is ",t_pre)
    # do LS:
    start = time()
    θ, stat = cgls(K, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    display(stat)
    # check MAE of training data only:
    errors = abs.(K*θ - E[Midx]) .* 627.503
    MAEtrain = sum(errors)/length(errors)
    # prediction (MOL GAUSS MODE!!):
    t_pred = @elapsed begin
        K_pred = get_norms(F, Widx, Midx)
        comp_gaussian_kernel!(K_pred, σ2)
        E_pred = K_pred*θ
    end
    # ATOM GAUSS MODE, need batch:
    #= Nqm9 = length(Widx)
    blength = Nqm9 ÷ bsize # number of batch iterations
    batches = kl_indexer(blength, bsize)
    bend = batches[end][end]
    bendsize = Nqm9 - (blength*bsize)
    push!(batches, bend+1 : bend + bendsize)
    t_pred = @elapsed begin
        K_pred = zeros(Nqm9, length(Midx))
        for batch in batches[1:end-1]
            #K_pred[batch, :] .= get_norms_at(F, Widx[batch], Midx)
            K_pred[batch, :] .= comp_gaussian_kernel_at(get_norms_at(F, Widx[batch], Midx), σ2)
        end
        #K_pred[batches[end], :] .= get_norms_at(F, Widx[batches[end]], Midx)
        K_pred[batches[end], :] .= comp_gaussian_kernel_at(get_norms_at(F, Widx[batches[end]], Midx), σ2)
        E_pred = K_pred*θ
    end =#
    errors = abs.(E_pred - E[Widx]) .* 627.503
    display(E_pred)
    MAE = sum(errors)/length(errors)
    println([σ0, σ2])
    println("pre-computation time is ",t_pre, ", MAEtrain=",MAEtrain)
    println("MAE of Nqm9 = ",MAE, ", with t_pred = ", t_pred)

    # for reduced energy, remove later!:
    fullidx = 1:n_data
    K = get_norms(F, fullidx, Midx)
    comp_gaussian_kernel!(K, σ2)
    E_pred = K*θ
    display(E_pred[Widx])
    return E_pred
end

function fitter_NN(F, E, Midx, Widx, foldername; Er = Vector{Float64}()::Vector{Float64})
    nK = length(Midx); Nqm9 = length(Widx); nf = size(F, 1)
    x_train = F[:, Midx]
    E_train = reduce(hcat, E[Midx])
    loader = Flux.DataLoader((x_train, E_train))

    x_test = F[:, Widx]

    # model:
    model = Chain(
        Dense(nf => 10, relu),   # activation function inside layer
        Dense(10 => 1)
        )
    pars = Flux.params(model)
    opt = Flux.Adam(0.01)

    # optimize:
    losses = []
    t = @elapsed begin
        @showprogress for epoch in 1:1_000
            for (x, y) in loader
                loss, grad = Flux.withgradient(pars) do
                    # Evaluate model and loss inside gradient context:
                    y_hat = model(x)
                    Flux.mse(y_hat, y)
                end
                Flux.update!(opt, pars, grad)
                push!(losses, loss)  # logging, outside gradient context
            end
        end
    end
    E_pred = vec(model(x_train))
    errors = abs.(E_pred - E[Midx]) .* 627.503
    MAE_train = sum(errors)/length(errors)
    display(MAE_train)

    # pred Nqm9:
    t_pred = @elapsed begin
        E_pred = vec(model(x_test))
    end
    if !isempty(Er) # check if the energy for fitting were reduced
        E_pred .+= Er[Widx] 
    end
    errors = abs.(E_pred - E[Widx]) .* 627.503
    MAE = sum(errors)/length(errors)
    println("pred Nqm9 MAE = ",MAE, " training time = ", t)
    # save info to file
    strlist = string.([MAE, Nqm9, nK, nf, t, t_pred])
    open("result/$foldername/err_$foldername.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    # clear memory:
    x_train=model=pars=opt=nothing; GC.gc()
end

function fit_NN(foldername)
    println("FITTING: $foldername")
    # input files:
    path = "data/$foldername/"
    mkpath("result/$foldername")
    file_dataset = path*"dataset.jld"
    file_finger = path*"features.jld"
    file_centers = path*"center_ids.jld"
    files = [file_dataset, file_finger, file_centers]
    dataset, F, Tidx = [load(f)["data"] for f in files]
    E = map(d -> d["energy"], dataset)
    # compute indices:
    n_data = length(dataset);
    K_indexer = 1:100 # 🌸 temporary selection
    Midx = Tidx[K_indexer] 
    Uidx = setdiff(Tidx, Midx) # (U)nsupervised data
    Widx = setdiff(1:n_data, Midx) # for evaluation 
    
    # data setup:
    F = F' # transpose since flux takes data in column
    nf = size(F, 1)
    x_train = F[:, Midx]
    E_train = reduce(hcat, E[Midx])
    loader = Flux.DataLoader((x_train, E_train))

    x_test = F[:, Widx]

    # model:
    model = Chain(
        Dense(nf => 10, relu),   # activation function inside layer
        Dense(10 => 1)
        )
    pars = Flux.params(model)
    opt = Flux.Adam(0.01)

    # optimize:
    losses = []
    t = @elapsed begin
        @showprogress for epoch in 1:1_000
            for (x, y) in loader
                loss, grad = Flux.withgradient(pars) do
                    # Evaluate model and loss inside gradient context:
                    y_hat = model(x)
                    Flux.mse(y_hat, y)
                end
                Flux.update!(opt, pars, grad)
                push!(losses, loss)  # logging, outside gradient context
            end
        end
    end
    display(losses)
    E_pred = vec(model(x_train))
    errors = abs.(E_pred - E[Midx]) .* 627.503
    display([E_pred, E[Midx]])
    MAE_train = sum(errors)/length(errors)
    display(MAE_train)

    # pred Nqm9:
    E_pred = vec(model(x_test))
    errors = abs.(E_pred - E[Widx]) .* 627.503
    MAE = sum(errors)/length(errors)
    display([E_pred, E[Widx]])
    println("pred Nqm9 MAE = ",MAE, " training time = ", t)
end

"""
atomic gaussian fitting (FCHL-ish)
"""
function fitter_GAK(F, f, dataset, E, Midx, Widx, foldername, tlimit; cσ = 2. * (2^5)^2, Er = Vector{Float64}()::Vector{Float64})
    nK = length(Midx); Nqm9 = length(Widx); 
    n_f = 0
    if !isempty(F) # could be empty since GAK only depends on atomic features
        n_f = size(F, 2)
    end
    # fit gausatom:
    #cσ = 2*(2^5)^2 # hyperparameter cσ = 2σ^2, σ = 2^k i guess
    A = get_gaussian_kernel(f[Midx], f[Midx], [d["atoms"] for d ∈ dataset[Midx]], [d["atoms"] for d ∈ dataset[Midx]], cσ)
    start = time()
    t_ls = @elapsed begin
        θ, stat = cgls(A, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    end
    #θ = A\E_train
    # check MAE of training data only:
    E_pred = A*θ # return the magnitude
    if !isempty(Er)
        E_pred .+= Er[Midx]
        E[Midx] .+= Er[Midx]
    end
    errors = E_pred - E[Midx]
    MAEtrain = mean(abs.(errors))*627.503 # in kcal/mol
    println("MAE train = ",MAEtrain)
    # pred ∀ data
    t = @elapsed begin
        A = get_gaussian_kernel(f[Widx], f[Midx], [d["atoms"] for d in dataset[Widx]], [d["atoms"] for d in dataset[Midx]], cσ)
    end
    println("kernel pred t = ",t)
    E_pred = A*θ
    if !isempty(Er)
        E_pred .+= Er[Widx]
    end 
    errors = E_pred - E[Widx]
    MAE = mean(abs.(errors)) * 627.503 # in kcal/mol
    println("MAE test = ",MAE)
    # save info to file
    strlist = string.([MAE, Nqm9, nK, n_f, t_ls, t])
    open("result/$foldername/err_$foldername.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    # clear variable
    A=θ=stat=E_pred=errors=nothing; GC.gc()
end

function fitter_repker(F, f, dataset, E, Midx, Widx, foldername, tlimit; Er = Vector{Float64}()::Vector{Float64})
    nK = length(Midx); Nqm9 = length(Widx); 
    n_f = 0
    if !isempty(F) # could be empty since GAK only depends on atomic features
        n_f = size(F, 2)
    end
    # fit gausatom:
    #cσ = 2*(2^5)^2 # hyperparameter cσ = 2σ^2, σ = 2^k i guess
    A = get_repker_atom(f[Midx], f[Midx], [d["atoms"] for d ∈ dataset[Midx]], [d["atoms"] for d ∈ dataset[Midx]])
    start = time()
    t_ls = @elapsed begin
        θ, stat = cgls(A, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    end
    # check MAE of training data only:
    E_pred = A*θ # return the magnitude
    if !isempty(Er)
        E_pred .+= Er[Midx]
        E[Midx] .+= Er[Midx]
    end
    errors = E_pred - E[Midx]
    MAEtrain = mean(abs.(errors))*627.503 # in kcal/mol
    println("MAE train = ",MAEtrain)
    # pred ∀ data
    t = @elapsed begin
        A = get_repker_atom(f[Widx], f[Midx], [d["atoms"] for d ∈ dataset[Widx]], [d["atoms"] for d ∈ dataset[Midx]])
    end
    println("kernel pred t = ",t)
    E_pred = A*θ
    if !isempty(Er)
        E_pred .+= Er[Widx]
    end 
    errors = E_pred - E[Widx]
    MAE = mean(abs.(errors)) * 627.503 # in kcal/mol
    println("MAE test = ",MAE)
    # save info to file
    strlist = string.([MAE, Nqm9, nK, n_f, t_ls, t])
    open("result/$foldername/err_$foldername.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    # clear variable
    A=θ=stat=E_pred=errors=nothing; GC.gc()
end

"""
fit the atomic energy for energy reducer
"""
function fit_atom(foldername, file_dataset, file_atomref_features; center_ids = [], tlimit = 900, uid="", kid="", save_global=false)
    mkpath("result/$foldername")
    dataset = load(file_dataset, "data"); n_data = length(dataset)
    F_atom = load(file_atomref_features, "data")
    E = map(d -> d["energy"], dataset)
    if isempty(center_ids)
        center_ids = load("data/$foldername/center_ids.jld", "data")[1]
    end
    K_indexer = 1:100 # 🌸 temporary selection
    Midx = center_ids[K_indexer] 
    Widx = setdiff(1:n_data, Midx)
    # compute atomic reference energies:
    A = F_atom[Midx, :] # construct the data matrix
    start = time()
    θ, stat = cgls(A, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    errors = A*θ - E[Midx]
    MAEtrain = mean(abs.(errors))*627.503
    println("atomic MAE train = ",MAEtrain)
    E_pred = F_atom[Widx, :]*θ
    errors = E_pred - E[Widx]
    MAE = mean(abs.(errors))*627.503 # the mean absolute value of the reduced energy but for the test set only
    println("atomic MAE test = ",MAE)
    E_atom = F_atom*θ # the sum of atomic energies
    E_red_mean = mean(abs.(E - E_atom)) .* 627.503 # mean of reduced energy
    # save MAE and atomref energies to file
    if isempty(uid)
        uid = readdlm("data/$foldername/setup_info.txt")[end, 1] # get data setup uid
    end
    if isempty(kid)
        kid = "K1"
    end
    strlist = string.(vcat(uid, kid, MAEtrain, E_red_mean, θ)) # concat the MAEs with the atomic ref energies
    open("result/$foldername/atomref_info.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    if save_global
        open("data/atomref_info.txt","a") do io # write also to global
            str = ""
            for s ∈ strlist
                str*=s*"\t"
            end
            print(io, str*"\n")
        end
    end
    # reduced energy data structure:
    Ed = Dict()
    Ed["theta"] = θ # or the atom reference energy
    Ed["atomic_energies"] = E_atom # sum of the atom ref energy
    save("result/$foldername/atom_energies.jld","data",Ed) # save also the reduced energy
    # clear memory:
    dataset=F_atom=E=center_ids=Midx=Widx=A=θ=stat=errors=E_pred=E_atom=E_red_mean=Ed=nothing
    GC.gc()
end

"""
this first fits the atomic reference energy, then fits model as usual using reduced energy
currently excludes the active training
"""
function fit_🌹_and_atom(foldername, file_dataset; 
                        bsize=1_000, tlimit=900, model="ROSEMI", 
                        E_atom=[], cσ = 2. *(2. ^5)^2, scaler=2. *(2. ^5)^2, center_ids=[], uid="", kid="")
    # file loaders:
    println("FITTING: $foldername")
    println("model type = ", model)
    # input files:
    path = "data/$foldername/"
    mkpath("result/$foldername")
    file_atom_E = "result/$foldername/atom_energies.jld" # atomic energies
    file_finger = path*"features.jld" # mol feature for model
    file_finger_atom = path*"features_atom.jld"
    file_spline = path*"spline.jld"
    file_dspline = path*"dspline.jld"
    files = [file_dataset, file_finger_atom, file_finger, file_spline, file_dspline]
    dataset, f, F, ϕ, dϕ = [load(f)["data"] for f in files] #F_atom is for fitting energy reducer, f is atomic features for the molecular fitting
    if isfile(file_atom_E) # separated, since this may be empty
        E_dict = load(file_atom_E, "data")
    end
    if isempty(center_ids)
        center_ids = load("data/$foldername/center_ids.jld", "data")[1]
    end
    D = fcenterdist(F, center_ids) # compute distances here, since it depends on the centers
    println("data loading finished!")
    F = F'; E = map(d -> d["energy"], dataset)
    # index computation:
    n_data = length(dataset); n_feature = size(F, 1);
    K_indexer = 1:100 # 🌸 temporary selection
    Midx = center_ids[K_indexer]
    Uidx = setdiff(center_ids, Midx) # (U)nsupervised data
    Widx = setdiff(1:n_data, Midx)
    # reduce training energy:
    if isempty(E_atom)
        E_atom = E_dict["atomic_energies"] # E_atom := E_null here
    end
    E[Midx] .-= E_atom[Midx] # a vector with length NQm9
    # write model header string:
    machine = splitdir(homedir())[end]; machine = machine=="beryl" ? "SAINT" : "OMP1" # machine name
    if isempty(uid)
        uid = readdlm("data/$foldername/setup_info.txt")[end, 1] # get data setup uid
    end
    if isempty(kid)
        kid = "K1"
    end
    strlist = string.(vcat(uid, kid, machine, model))
    open("result/$foldername/err_$foldername.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str)
    end
    # model fitting:
    println("fitting starts!")
    if model=="ROSEMI"
        MAE, MADmax_idxes = fitter(F, E, D, ϕ, dϕ, Midx, center_ids, Uidx, Widx, n_feature, foldername, bsize, tlimit; Er = E_atom)
    elseif model == "KRR" 
        fitter_KRR(F', E, Midx, center_ids, Widx, K_indexer, foldername, tlimit, n_feature; Er = E_atom, scaler=scaler)
    elseif model == "NN"
        fitter_NN(F, E, Midx, Widx, foldername; Er = E_atom) # no tlimit yet, but mostly dont really matter
    elseif model == "LLS"
        fitter_LLS(F', E, Midx, Widx, foldername, tlimit; Er = E_atom)
    elseif model == "GAK" # atomic model
        fitter_GAK(F', f, dataset, E, Midx, Widx, foldername, tlimit; cσ=cσ, Er = E_atom) # takes atomic features instead
    elseif model == "REAPER" # atomic model
        fitter_repker(F', f, dataset, E, Midx, Widx, foldername, tlimit; Er = E_atom)
    end
    # clear memory:
    dataset = E_dict = f = F = ϕ = dϕ = E = D = E_atom = Midx = Uidx = Widx = nothing; GC.gc()
end

"""
automatically generate data and fit based on list of molname, n_data, n_feature,M, and universe_size saved in json file 
"""
function autofit_🌹() 
    json_string = read("setup.json", String)
    d = JSON3.read(json_string)
    molnames = d["mol_name"]; ndatas = d["n_data"]; nfeatures = d["n_feature"]; Ms = d["M"] # later the all of the other vars should be a list too!, for now assume fixed
    for i ∈ eachindex(molnames)
        data_setup(molnames[i], ndatas, nfeatures, Ms) # setup data
        fit_🌹(molnames[i], ndatas, nfeatures, Ms) # fit data!
    end
end

"""
just a quick pred func
"""
function predict(mol_name, n_data, n_feature, M)
    res = load("result/H7C8N1/theta_center_$mol_name"*"_[100, 150, 24, 8].jld")["data"] # load optimized parameters
    θ = res["theta"]
    colsize = length(θ)
    # load required data:
    path = "data/$mol_name/"
    file_dataset = path*"$mol_name"*"_dataset_$n_data.jld"
    file_finger = path*"$mol_name"*"_ACSF_$n_feature"*"_$n_data.jld"
    file_distance = path*"$mol_name"*"_distances_$n_feature"*"_$n_data.jld"
    file_centers = path*"$mol_name"*"_M=$M"*"_$n_feature"*"_$n_data.jld"
    # setup parameters:
    n_basis = 5 # pre-inputted number, later n_basis := n_basis+3 🌸
    dataset = load(file_dataset)["data"] # energy is from here
    W = load(file_finger)["data"]' # load and transpose the normalized fingerprint (sometime later needs to be in feature × data format already so no transpose)
    s_W = size(W) # n_feature × n_data
    n_feature = s_W[1]; n_data = s_W[2]; data_idx = 1:n_data
    Midx = load(file_centers)["data"]; Midx = Midx[1:100] # slice the M indexes!!
    Widx = setdiff(data_idx, Midx)
    E = map(d -> d["energy"], dataset)
    #println(E)
    D = load(file_distance)["data"] # the mahalanobis distance matrix
    # index op:
    ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true) # compute basis from fingerprint ∈ (n_feature*(n_basis+3), n_data)
    n_basis += 3 # by definition of bspline
    M = length(Midx); N = length(Widx); L = n_feature*n_basis; row = M*N; col = M*L
    # precompute stuffs:
    SKs = map(m -> comp_SK(D, Midx, m), Widx)
    γ = comp_γ(D, SKs, Midx, Widx)
    α = γ .- 1
    B = zeros(N, M*L); 
    comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature);
    klidx = kl_indexer(M, L)
    cidx = 1:M
    # get MAE and MAD:
    v = zeros(row); vmat = zeros(N, M); VK = zeros(N); tempsA = [zeros(N) for _ = 1:7] # replace temp var for memefficiency
    comp_v!(v, vmat, VK, tempsA, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α) # compute errors
    ΔEs = abs.(VK .- E[Widx]) .* 627.503 # convert from Hartree to kcal/mol
    MAE = sum(ΔEs) / N
    MADs = (vec(sum(abs.(vmat), dims=2)) ./ M) .* 627.503 # vector of length N and convert to kcal/mol
    println("MAE of all mol w/ unknown E is ", MAE)
    sidx = sortperm(ΔEs); madidx = sortperm(MADs)
    Wstr = string.(Widx)
    p1 = scatter(Wstr[sidx], ΔEs[sidx], xlabel = L"$m$", ylabel = L"$\Delta E_m$ (kcal/mol)", legend = false)
    p2 = scatter(Wstr[sidx], log10.(MADs[sidx]), xlabel = L"$m$", ylabel = L"log$_{10}$MAD($w_m$) (kcal/mol)", legend = false)
    display(p1); savefig(p1, "plot/Delta_E_new.png")
    display(p2); savefig(p2, "plot/MAD_new.png")
    println("MAE of all mol w/ unknown E is ", MAE)
end


"""
self use only, # save this to PDF
"""
function check_MAE()
    foldername = "exp_5k"
    path = "data/$foldername/"
    file_dataset = path*"dataset.jld"
    file_finger = path*"features.jld"
    file_distance = path*"distances.jld"
    file_centers = path*"center_ids.jld"
    files = [file_dataset, file_finger, file_distance, file_centers]
    dataset, F, D, T = [load(f)["data"] for f in files]
    E = map(d -> d["energy"], dataset)
    K = T[1:100]
    E_med = median(E[K])
    MAE_approx = sum(abs.(E .- E_med))
    display(MAE_approx)
end

function test_FCHL()
    dataset = load("data/qm9_dataset_old.jld", "data")
    f = load("data/FCHL.jld", "data")
    centers = vec(readdlm("data/centers.txt", Int))
    E_null = vec(readdlm("data/atomic_energies.txt"))
    K = centers[1:100]
    A = comp_FCHL_kernel_entry(f[1, :, :, :], f[1, :, :, :], dataset[1]["atoms"], dataset[1]["atoms"], 32.)
    display(A[1:5, 1:5])
end

"""

"""
function center_comp_driver()
    limits = [51, 165]
    nfpercent = [100, 80, 75, 60, 50, 40, 20] #% of features brought
    data_setup("exp_reduced_energy", naf, nmf, 2, 300, "data/qm9_dataset_old.jld",
                ff, fname; save_global_centers = true)
end

"""
fit all NQM9 data for null model, separate function since the default function can't accomodate full data (error)
"""
function fit_all_null(foldername, file_dataset, file_atomref_features; center_ids = [], tlimit = 900, uid="", kid="", save_global=false)
    mkpath("result/$foldername")
    dataset = load(file_dataset, "data"); n_data = length(dataset)
    F_atom = load(file_atomref_features, "data")
    E = map(d -> d["energy"], dataset)
    if isempty(center_ids)
        center_ids = load("data/$foldername/center_ids.jld", "data")[1]
    end
    Midx = center_ids
    # compute atomic reference energies:
    A = F_atom[Midx, :] # construct the data matrix
    start = time()
    θ, stat = cgls(A, E[Midx], itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit))
    errors = A*θ - E[Midx]
    MAEtrain = mean(abs.(errors))*627.503
    println("atomic MAE train = ",MAEtrain)
    E_atom = F_atom*θ # the sum of atomic energies
    E_red_mean = mean(abs.(E - E_atom)) .* 627.503 # mean of reduced energy
    # save MAE and atomref energies to file
    if isempty(uid)
        uid = readdlm("data/$foldername/setup_info.txt")[end, 1] # get data setup uid
    end
    if isempty(kid)
        kid = "K1"
    end
    strlist = string.(vcat(uid, kid, MAEtrain, E_red_mean, θ)) # concat the MAEs with the atomic ref energies
    open("result/$foldername/atomref_info.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    if save_global
        open("data/atomref_info.txt","a") do io # write also to global
            str = ""
            for s ∈ strlist
                str*=s*"\t"
            end
            print(io, str*"\n")
        end
    end
    # reduced energy data structure:
    Ed = Dict()
    Ed["theta"] = θ # or the atom reference energy
    Ed["atomic_energies"] = E_atom # sum of the atom ref energy
    save("result/$foldername/atom_energies.jld","data",Ed) # save also the reduced energy
end


"""
======== ΔML stuffs ============
"""

"""
computes E_db as the highest level, given a set of training indices
"""
function get_delta_Edb(E, Fda, Fdb, idtrain, idtest)
    # fit the baselines, dressed_atom and dressed_bonds:
    # vector structure for each set: [MAE_train_eda, MAE_test_eda, MAE_train_edb, MAE_test_edb]
    MAEs = []
    θ = Fda[idtrain, :]\E[idtrain]
    Eda = Fda*θ
    MAEtrain = mean(abs.(E[idtrain] - Eda[idtrain]))*627.503
    MAEtest = mean(abs.(E[idtest] - Eda[idtest]))*627.503
    println("dressed_atom: ", MAEtrain, " ",MAEtest)
    push!(MAEs, MAEtrain); push!(MAEs, MAEtest)
    # dressed bonds:
    Et = E - Eda # take out parts of the energy
    θ = Fdb[idtrain, :]\Et[idtrain]
    Edb = Fdb*θ 
    MAEtrain = mean(abs.(Et[idtrain] - Edb[idtrain]))*627.503
    MAEtest = mean(abs.(Et[idtest] - Edb[idtest]))*627.503
    println("dressed_bonds: ", MAEtrain, " ",MAEtest)
    push!(MAEs, MAEtrain); push!(MAEs, MAEtest)
    return E-Eda-Edb, MAEs # return the vector of MAEs and the vector of energies
end


"""
try out fitting with current best found feature and current best found model
WITHOUT data selection for: Ebase = nothing, Ebase = NullModel, Ebase = SoB

now rerun with all the new features (large sparse ones) and filtered dataset, save the MAEs in table, therefore:
table rows = features × models × solver × n_splits
cols (headers) = header(rows) ∪ {MAEtrain, MAEtest, Elevel×solver}
"""
function test_DeltaML()
    # def:
    E = readdlm("data/energies.txt")
    nrow = length(E)
    
    # select split indexes, will be used for baseline and last level fitting:
    Random.seed!(603)
    idall = 1:nrow
    idtrain = sample(1:nrow, 100, replace=false)
    idtest = setdiff(idall, idtrain)
    
    # fit the baselines, dressed_atom and dressed_bonds:
    MAEs = Matrix{Any}(undef, 5,4)
    MAEs[1,:] = ["Elevel", "solver", "MAEtrain", "MAEtest"]; 
    MAEs[2:5, 1] = ["dressed_atom", "dressed_atom", "dressed_bond", "dressed_bond"] # MAEs of base models
    MAEs[[2,4],2] = ["direct", "direct"]; MAEs[[3,5],2] = ["CGLS", "CGLS"]
    # dressed atom:
    F = load("data/atomref_features.jld", "data")
    θ1 = F[idtrain, :]\E[idtrain]; θ2, stat = cgls(F[idtrain, :], E[idtrain], itmax=500)
    Edas = []
    push!(Edas, F*θ1)
    MAEs[2,3] = mean(abs.(E[idtrain] - Edas[1][idtrain]))*627.503
    MAEs[2,4] = mean(abs.(E[idtest] - Edas[1][idtest]))*627.503
    push!(Edas, F*θ2)
    MAEs[3,3] = mean(abs.(E[idtrain] - Edas[2][idtrain]))*627.503
    MAEs[3,4] = mean(abs.(E[idtest] - Edas[2][idtest]))*627.503
    println("dressed_atom: ", MAEs[2:3, 3:4])
    Eda = Edas[argmin(MAEs[2:3,4])] # save dressed atom energies with the lowest MAE
    # dressed bonds:
    F = load("data/featuresmat_qm9_covalentbonds.jld", "data")
    Et = E - Eda # take out parts of the energy
    θ1 = F[idtrain, :]\Et[idtrain]; θ2, stat = cgls(F[idtrain, :], Et[idtrain], itmax=500)
    Edbs = []
    push!(Edbs, F*θ1)
    MAEs[4,3] = mean(abs.(Et[idtrain] - Edbs[1][idtrain]))*627.503
    MAEs[4,4] = mean(abs.(Et[idtest] - Edbs[1][idtest]))*627.503
    push!(Edbs, F*θ2)
    MAEs[5,3] = mean(abs.(Et[idtrain] - Edbs[2][idtrain]))*627.503
    MAEs[5,4] = mean(abs.(Et[idtest] - Edbs[2][idtest]))*627.503
    println("dressed_atom: ", MAEs[4:5, 3:4])
    Edb = Edbs[argmin(MAEs[4:5,4])] # save dressed bond energies with the lowest MAE
    writedlm("result/deltaML/MAE_base.txt", MAEs)
    writedlm("data/energy_clean_db.txt", E-Eda-Edb) # save cleaned energy


    # test diverse models: check TRAIN first for correctness
    features = ["ACSF_51", "SOAP", "FCHL19"] # outtest loop
    models = ["LLS", "GAK", "REAPER"][2:3]
    solvers = ["direct", "cgls"]
    elvs = ["dressed_atom", "dressed_bond"]
    n_trains = [10, 25, 50, 100] # ni+1 = 2ni, max(ni) = 100; innest loop
    outs = Matrix{Any}(undef, length(features)*length(models)*length(n_trains)*length(solvers)*length(elvs) + 1, 7) # output table
    outs[1,:] = ["ntrain", "feature", "model", "solver", "Elevel", "MAEtrain", "MAEtest"]
    # enumerate (cartesian product):
    iters = Iterators.product(n_trains, solvers, models, elvs)
    cr = 2
    dataset = load("data/qm9_dataset.jld", "data")
    σ = 2048.
    for feat ∈ features
        f = load("data/"*feat*".jld", "data")
        # compute all kernels here once per feature to save computation time:
        println("computing kernels...")
        t = @elapsed begin
            Kg = get_gaussian_kernel(f, f[idtrain], [d["atoms"] for d ∈ dataset], [d["atoms"] for d ∈ dataset[idtrain]], σ)
            Kr = get_repker_atom(f, f[idtrain], [d["atoms"] for d ∈ dataset], [d["atoms"] for d ∈ dataset[idtrain]]) 
        end
        println("kernel computation is finished in ",t)
        for it ∈ iters
            n = it[1]; solver = it[2]; model = it[3]; lv = it[4]
            println(it, " ",feat)
            # indexes:
            idtr = idtrain[1:n]
            idts = setdiff(idall, idtr)
            ktrid = indexin(idtr, idall) # kernel relative train index, these indexes are needed if the feature vector is sliced
            ktsid = indexin(idts, idall) # kernel relative test index
            # Elevel:
            if lv == "dressed_atom"
                Et = E-Eda
            elseif lv == "dressed_bond"
                Et = E-Eda-Edb
            end
            # model train:
            if model == "GAK"
                K = Kg[ktrid, :] 
            elseif model == "REAPER"
                K = Kr[ktrid, :]
            end
            # solver:
            if solver == "direct"
                K[diagind(K)] .+= 1e-8
                θ = K\Et[idtr]
            elseif solver == "cgls"
                θ, stat = cgls(K, Et[idtr], itmax=500, λ = 1e-8)
            end
            Epred = K*θ
            MAEtrain = mean(abs.(Et[idtr] - Epred))*627.503
            # model test:
            if model == "GAK"
                K = Kg[ktsid, :] 
            elseif model == "REAPER"
                K = Kr[ktsid, :] 
            end
            Epred = K*θ
            MAE = mean(abs.(Et[idts] - Epred))*627.503
            # data output:
            outs[cr, 1] = n; outs[cr, 2] = feat; outs[cr, 3] = model; 
            outs[cr, 4] = solver; outs[cr, 5] = lv; outs[cr, 6] = MAEtrain; outs[cr, 7] = MAE 
            println(outs[cr, :], "done !")
            open("result/deltaML/MAE_enum.txt", "a") do io # writefile by batch
                writedlm(io, permutedims(outs[cr,:]))
            end
            cr += 1
        end
    end
    display(outs)
end

"""
test data selection given a feature WITHOUT PCA,

"""
function test_selection_delta()
    # Scenario 1: compute centers(feature) -> 5 sets of centers ∀features, get the set of centers with the lowest MAE(Edb):
    dataset = load("data/qm9_dataset.jld", "data")
    features = ["ACSF_51", "SOAP", "FCHL19"]
    all_centers = []
    for feat ∈ features
        println(feat)
        f = load("data/"*feat*".jld", "data")
        tF = @elapsed begin
            F = extract_mol_features(f, dataset)[:, 1:end-5] 
        end
        println("moltransform elapsed = ", tF)
        display(F)
        tC = @elapsed begin
            centers = set_cluster(F, 200; universe_size = 1000, num_center_sets = 5)
        end
        println("selection elapsed = ", tC)
        centers = reduce(hcat, centers)
        if isempty(all_centers)
            all_centers = centers
        else
            all_centers = hcat(all_centers, centers)
        end
    end
    writedlm("data/all_centers_deltaML.txt", all_centers)
    # Scenario 2: get the lowest MAE(Edb) for each feature type from the above sets
end

function test_get_lowest_MAE()
    E = readdlm("data/energies.txt")
    all_centers = Int.(readdlm("data/all_centers_deltaML.txt")[1:100, :])
    idall = 1:length(E)
    Fda = load("data/atomref_features.jld", "data")
    Fdb = load("data/featuresmat_qm9_covalentbonds.jld", "data")
    MAE_tb = []
    for i ∈ axes(all_centers, 2)
        idtrain = all_centers[:, i]
        idtest = setdiff(idall, idtrain)
        # compute Edb:
        E_clean, MAEs = get_delta_Edb(E, Fda, Fdb, idtrain, idtest)
        push!(MAE_tb, MAEs)
    end
    MAE_tb = reduce(vcat, MAE_tb')
    display(MAE_tb)
end

"""
test PyCall for QML
"""
#= function qml(cutoff)
    py"""
    import numpy as np
    from os import listdir, makedirs
    from os.path import isfile, join, exists
    import time
    from warnings import catch_warnings

    import qml
    from qml.fchl import generate_representation, get_local_kernels, get_atomic_kernels, get_atomic_symmetric_kernels
    from qml.math import cho_solve
    
    def fchl(cutoff):
        print(cutoff)
        fpath = "data/qm9_error.txt"
        with open(fpath,'r') as f: # errorlist
            strlist = f.read()
            strlist = strlist.split("\n")
            errfiles = strlist[:-1]

        mypath = "data/qm9/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and (f not in errfiles)] # remove errorfiles
        onlyfiles = sorted(onlyfiles) #[1:] this is for nonubuntu
        print(len(onlyfiles), onlyfiles[0])

        # extract coords here:
        start = time.time() # timer
        # make FCHL folder
        feature_folder = "data/FCHL"
        if not exists(feature_folder):
            makedirs(feature_folder)
        
        n_atom_QM9 = 29
        for f in sorted(onlyfiles)[:5]:
            # extract features:
            mol = qml.Compound(xyz=mypath+f)#, qml.Compound(xyz="data/qm9/dsgdb9nsd_000002.xyz")
            mol.generate_fchl_representation(max_size=n_atom_QM9, cut_distance=cutoff, neighbors=n_atom_QM9) # neighbours is only used if it has periodic boundary
            print(mol.name)
    """
    py"fchl"(cutoff)

end =#

"""
unused stuffs but probably needed later..
"""
function junk()
    # fit, try lsovle vs lsquares!:
    #= θ = rand(Uniform(-1., 1.), size(A)[2]) # should follow the size of A, since actual sparse may not reach the end of index # OLD VER: θ = rand(Uniform(-1., 1.), cols)
    function df!(g, θ) # closure function for d(f_obj)/dθ
        g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
    end
    res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS(m=1_000), Optim.Options(show_trace=true, iterations=1_000))
    θ = Optim.minimizer(res)
    println(res) =#

    #= # linear solver:
    t = @elapsed begin
        θ = A\b
    end
    println("lin elapsed time: ", t)
    println("lin obj func = ", lsq(A, θ, b)) =#

     #= i = 1; j = Midx[i]; m = Widx[1]
    ΔjK = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=false)
    ΔjK_m = comp_ΔjK_m(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=false)
    display([r[i], A[i,:]'*θ - b[i], ΔjK, ΔjK_m]) # the vector slicing by default is column vector in Julia! =#
        
end
