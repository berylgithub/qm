using Krylov, LsqFit, ReverseDiff, ForwardDiff, BenchmarkTools, Optim, Printf, JSON3, DelimitedFiles
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
function set_cluster(F::Matrix{Float64}, M; universe_size=1_000)
    N, L = size(F)
    F = F'
    idx = 603 # the ith data point of the dataset, can be arbitrary technically, for now fix 603:= RoZeMi 🌹
    idx = Int(round(idx/universe_size*N)) # relative idx
    wbar, C = mean_cov(F, idx, N, L)
    B = Matrix{Float64}(I, L, L) # try with B = I #B = compute_B(C) 
    # generate centers (M) for training:
    center_ids, mean_point, D = eldar_cluster(F, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd", get_distances=true) # generate cluster centers
    return center_ids, D
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
"""
function data_setup(foldername, data_indices, n_af, n_mf, n_basis, num_centers, feature_file; universe_size=1_000)
    println("data setup for n_data = ",length(data_indices),", atom features = ",n_af, ", mol features = ", n_mf, ", centers = ",num_centers, " starts!")
    t = @elapsed begin
        path = mkpath("data/$foldername")
        # slice:
        dataset = load("data/qm9_dataset.jld")["data"]
        dataset = dataset[data_indices]
        display(dataset)
        # PCA:
        F = load(feature_file)["data"] # pre-extracted features
        F = feature_extractor(F, n_af, n_mf)
        F = F[data_indices, :]
        # compute bspline:
        ϕ, dϕ = extract_bspline_df(F', n_basis; flatten=true, sparsemat=true) # move this to data setup later
        display(F)
        # get centers:
        center_ids, distances = set_cluster(F, num_centers, universe_size=universe_size)
        display(center_ids)
        # save files:
    end
    save("data/$foldername/dataset.jld", "data", dataset)
    save("data/$foldername/features.jld", "data", F)
    save("data/$foldername/center_ids.jld", "data", center_ids)
    save("data/$foldername/distances.jld", "data", distances)
    save("data/$foldername/spline.jld", "data", ϕ)
    save("data/$foldername/dspline.jld", "data", dϕ)
    println("data setup is finished in ",t,"s")
end

"""
main fitter function, assemble LS -> fit -> save to file
to avoid clutter in main function, called within fitting iters

outputs:
    - indexes of n-maximum MAD
"""
function fitter(F, E, D, ϕ, dϕ, Midx, Tidx, Uidx, Widx, n_feature, mol_name, bsize; get_mad=false)
    N = length(Tidx); nU = length(Uidx); nK = length(Midx); Nqm9 = length(Widx)
    nL = size(ϕ, 1); n_basis = nL/n_feature
    println(typeof(Nqm9))
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
        θ, stat = cgls(op, b, itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, 20)) # with callback 🌸
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
    display(MADs)
    # get MAE of test set (QM9):
    # BATCHMODE:
    #= MAE = sum(abs.(VK .- E[Widx])) / N
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
    matsize = [nK, nU, n_feature, n_basis]
    strlist = vcat(string.(matsize), [lstrip(@sprintf "%16.8e" s) for s in data], string(get_mad), string.([t_ab, t_ls]))
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
    return MAE, MADmax_idxes =#
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
function fit_🌹(foldername, bsize; mad = false)
    println("FITTING: $foldername")
    # input files:
    path = "data/$foldername/"
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
        MAE, MADmax_idxes = fitter(F, E, D, ϕ, dϕ, Midx, Tidx, Uidx, Widx, n_feature, foldername, bsize) #
    end
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
