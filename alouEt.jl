using Krylov, LsqFit, ReverseDiff, ForwardDiff, BenchmarkTools, Optim, Printf, JSON3, DelimitedFiles
"""
contains all tests and experiments
"""

include("voronoi.jl")
include("linastic.jl")
include("RoSemi.jl")


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
    display(A)
    println(N, " ", L)
    # compute mean and cov:
    idx = 603 # the ith data point of the dataset, can be arbitrary technically, for now fix 603:= RoZeMi 🌹
    idx = Int(round(idx/universe_size*N)) # relative idx
    wbar, C = mean_cov(A, idx, N, L)
    B = Matrix{Float64}(I, L, L) # try with B = I #B = compute_B(C) 
    display(wbar)
    display(B)
    # generate centers (M) for training:
    center_ids, mean_point = eldar_cluster(A, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd") # generate cluster centers
    display(mean_point)
    display(center_ids)
    return center_ids
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
        center_ids = set_cluster(main_file, M_actual, universe_size=universe_size)
        save(path*"/$mol_name"*"_M=$M"*"_$n_feature"*"_$n_data.jld", "data", center_ids)
        # compute all distances:
        Dist, idx = set_all_dist(main_file, universe_size=universe_size)
        save(path*"/$mol_name"*"_distances_"*"$n_feature"*"_$n_data.jld", "data", Dist)
        # scale feature for basis:
        #= W = normalize_routine(main_file)
        save(path*"/$mol_name"*"_ACSF_"*"$n_feature"*"_"*"$n_data"*"_symm_scaled.jld", "data", W) =#
    end
    println("data setup for mol=",mol_name,", n_data=", n_data,", n_feature=",n_feature,", M=", M, " is finished in $t seconds!!")
end

"""
main fitter function, assemble LS -> fit -> save to file
to avoid clutter in main function, called within fitting iters

outputs:
    - indexes of n-maximum MAD
"""
function fitter(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis, mol_name; get_mad=false)
    M = length(Midx); N = length(Widx); L = n_feature*n_basis
    row = M*N; col = M*L
    println("[M, N] = ",[M, N])
    #= t_ab = @elapsed begin
        # assemble A and b:
        A, b = assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis) #A, b = assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    end
    println("LS assembly time: ",t_ab)
    display(A)
    println("memory size of A = ", Base.summarysize(A)*1e-6, " megabytes")
    n_l = n_basis*n_feature # length of feature*basis each k
    # iterative linear solver (CGLS):
    t_ls = @elapsed begin
        linres = Krylov.cgls(A, b, itmax=500, history=true)  # 🌸
    end
    θ = linres[1]
    obj = lsq(A, θ, b)
    println("CGLS obj = ",obj, ", CGLS time = ",t_ls) =#

    # !!!! using LinearOperators !!!:
    # precompute stuffs:
    t_ab = @elapsed begin
        SKs = map(m -> comp_SK(D, Midx, m), Widx)
        γ = comp_γ(D, SKs, Midx, Widx)
        α = γ .- 1
        B = zeros(N, M*L); 
        comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature);
        klidx = kl_indexer(M, L)
        cidx = 1:M
    end
    println("precomputation time = ",t_ab)
    t_ls = @elapsed begin
        # generate LinOp in place of A!:
        Axtemp = zeros(N, M); tempsA = [zeros(N) for _ in 1:3]
        op = LinearOperator(Float64, row, col, false, false, (y,u) -> comp_Ax!(y, Axtemp, tempsA, u, B, Midx, cidx, klidx, γ, α), 
                                                            (y,v) -> comp_Aᵀv!(y, v, B, Midx, Widx, γ, α, L))
        show(op)
        # generate b:
        b = zeros(N*M); btemp = zeros(N, M); tempsb = [zeros(N) for _ in 1:2]
        comp_b!(b, btemp, tempsb, E, γ, α, Midx, cidx)
        # do LS:
        θ, stat = cgls(op, b, itmax=500, verbose=0) # 🌸
    end
    # get residual:
    obj = norm(op*θ - b)^2
    println("solver obj = ",obj, ", solver time = ",t_ls)

    # get MAE and MAD:
    v = zeros(row); vmat = zeros(N, M); VK = zeros(N); tempsA = [zeros(N) for _ = 1:7] # replace temp var for memefficiency
    comp_v!(v, vmat, VK, tempsA, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α)
    MAE = sum(abs.(VK .- E[Widx])) / N
    MADs = vec(sum(abs.(vmat), dims=2)) ./ M # length N
    MAE *= 627.503 # convert from Hartree to kcal/mol
    println("MAE of all mol w/ unknown E is ", MAE)
    # get the n-highest MAD:
    n = 1 # 🌸
    sidxes = sortperm(MADs)[end-(n-1):end]
    MADmax_idxes = Widx[sidxes] # the indexes relative to Widx (global data index)
    
    # get min |K| RMSD (the obj func):
    RMSD = obj #Optim.minimum(res)
    
    println("largest MAD is = ", MADs[sidxes[end]], ", with index = ",MADmax_idxes)
    println("min K|∑RMSD(w) = ", RMSD)

    # save all errors foreach iters:
    data = [MAE, RMSD, MADs[sidxes[end]]]
    matsize = [M, N, n_feature, n_basis]
    strlist = vcat(string.(matsize), [lstrip(@sprintf "%16.8e" s) for s in data], string(get_mad), string.([t_ab, t_ls]))
    open("result/$mol_name/err_$mol_name.txt","a") do io
        str = ""
        for s ∈ strlist
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
    # save also the M indices and θ's to file!!:
    data = Dict("centers"=>Midx, "theta"=>θ)
    save("result/$mol_name/theta_center_$mol_name"*"_$matsize.jld", "data", data)
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
    for i ∈ [10] # change iters here
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
    res = load("result/H7C8N1/theta_center_H7C8N1.jld")["data"] # load optimized parameters
    θ = res["theta"]
    colsize = length(θ)
    # load required data:
    path = "data/$mol_name/"
    file_dataset = path*"$mol_name"*"_dataset_$n_data.jld"
    file_finger = path*"$mol_name"*"_ACSF_$n_feature"*"_$n_data.jld"
    file_distance = path*"$mol_name"*"_distances_$n_feature"*"_$n_data.jld"
    file_centers = path*"$mol_name"*"_M=$M"*"_$n_feature"*"_$n_data.jld"
    # setup parameters:
    n_basis = 3 # pre-inputted number, later n_basis := n_basis+3 🌸
    dataset = load(file_dataset)["data"] # energy is from here
    W = load(file_finger)["data"]' # load and transpose the normalized fingerprint (sometime later needs to be in feature × data format already so no transpose)
    s_W = size(W) # n_feature × n_data
    n_feature = s_W[1]; n_data = s_W[2];
    E = map(d -> d["energy"], dataset)
    #println(E)
    D = load(file_distance)["data"] # the mahalanobis distance matrix
    # index op:
    data_idx = 1:n_data
    ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true) # compute basis from fingerprint ∈ (n_feature*(n_basis+3), n_data)
    n_basis += 3 # by definition of bspline
    n_l = n_basis*n_feature
    # setup matrix A:
    Midx = res["centers"]
    j = Midx[1] # arbitrary for MAE, dosent matter
    Widx = setdiff(data_idx, Midx)
    A, b = assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis) #A, b = assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    # compute MAE:
    ΔEs = zeros(length(Widx))
    c = 1
    for m ∈ Widx
        _, VK = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=true)
        err = abs(VK - E[m])
        ΔEs[c] = err
        c += 1
    end
    ΔEs *= 627.5
    sidx = sortperm(ΔEs)
    MAE = (sum(ΔEs)/length(Widx)) # convert from Hartree to kcal/mol
    Wstr = string.(Widx)
    p = scatter(Wstr[sidx], ΔEs[sidx], xlabel = L"$m$", ylabel = L"$\Delta E_m$ (kcal/mol)", legend = false)
    display(p)
    savefig(p, "plot/Delta_E.png")
    println("MAE of all mol w/ unknown E is ", MAE)
    display(lsq(A, θ, b))
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