using LsqFit, ReverseDiff, ForwardDiff, BenchmarkTools, Optim
"""
contains all tests and experiments
!!! FOR LATER: https://stackoverflow.com/questions/57950114/how-to-efficiently-initialize-huge-sparse-arrays-in-julia
"""

include("voronoi.jl")
include("linastic.jl")
include("RoSemi.jl")


"""
get the indices of the supervised datapoints M, fix w0 as i=603 for now
"""
function set_cluster()
    # load dataset || the datastructure is subject to change!! since the data size is huge
    dataset = load("data/ACSF_1000_symm.jld")["data"]
    N, L = size(dataset)
    A = dataset' # transpose data (becomes column major)
    display(A)
    println(N, " ", L)
    M = 10 # number of selected data
    # compute mean and cov:
    idx = 603 # the ith data point of the dataset, can be arbitrary technically
    wbar, C = mean_cov(A, idx, N, L)
    B = compute_B(C)
    display(wbar)
    display(B)
    # generate centers (M) for training:
    center_ids, mean_point = eldar_cluster(A, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd") # generate cluster centers
    display(mean_point)
    display(center_ids)
    # save center_ids:
    save("data/M=$M"*"_idx_$N.jld", "data", center_ids)
end


"""
compute all D_k(w_l) ∀k,l, for now fix i = 603 (rozemi)
"""
function set_all_dist()
    dataset = load("data/ACSF_1000_symm.jld")["data"]
    N, L = size(dataset)
    W = dataset' # transpose data (becomes column major)
    println(N, " ", L)
    M = 10 # number of selected supervised data
    # compute mean and cov:
    idx = 603 # the ith data point of the dataset, can be arbitrary technically (for now fix i=603)
    wbar, C = mean_cov(W, idx, N, L)
    B = compute_B(C)
    #display(wbar)
    #dist = f_distance(B, A[:,1], A[:,2])
    #display(dist)

    # compute all distances:
    filename = "data/distances_1000_i=603.jld"
    D = compute_distance_all(W, B, filename)
    display(D)
end


"""
the main fitting function !!!
targets: chemical accuracy = 1 kcal/mol = 0.0015936 Ha = 0.0433641 eV.
try:
    - use fixed N = 100
    - varying M (recomputing centers)
    - changing column length
    - multirestart
"""
function fit_rosemi()
    n_basis = 10 # pre-inputted number, later n_basis := n_basis+3
    dataset = load("data/qm9_dataset_1000.jld")["data"] # energy is from here
    W = load("data/ACSF_1000_symm_scaled.jld")["data"]' # load and transpose the normalized fingerprint (sometime later needs to be in feature × data format already so no transpose)
    s_W = size(W) # n_feature × n_data
    n_feature = s_W[1]; n_data = s_W[2];
    E = map(d -> d["energy"], dataset)
    D = load("data/distances_1000_i=603.jld")["data"] # the mahalanobis distance matrix
    # index op:
    data_idx = 1:n_data
    Midx = load("data/M=10_idx_1000.jld")["data"] # the supervised data points' indices
    n_m = size(Midx) # n_sup_data
    Widx = setdiff(data_idx, Midx) # the (U)nsupervised data, which is ∀i w_i ∈ W \ K, "test" data
    Widx = Widx[1:10] # take subset for smaller matrix
    #display(dataset)
    n_m = length(Midx); n_w = length(Widx)
    display([length(data_idx), n_m, n_w])
    
    ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true) # compute basis from fingerprint ∈ (n_feature*(n_basis+3), n_data)
    n_basis += 3 # by definition of bspline
    #display(ϕ)
    #display([nnz(ϕ), nnz(dϕ)]) # only ≈1/3 of total entry is nnz
    #display(Base.summarysize(ϕ)) # turns out only 6.5mb for sparse
    # assemble A and b:
    
    # === start fitting loop ===:
    loop_idx = 1:2
    for i ∈ loop_idx
        println("======= LOOP i=$i =======")
        t = @elapsed begin
            A, b = assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis) #A, b = assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
        end
        println("LS assembly time: ",t)
        #A = sparse(A) # only half is filled!!
        display(A)
        #display(b)

        # fit, try lsovle vs lsquares!:
        n_l = n_basis*n_feature # length of feature*basis each k
        cols = n_m*n_l # length of col
        θ = rand(Uniform(-1., 1.), size(A)[2]) # should follow the size of A, since actual sparse may not reach the end of index # OLD VER: θ = rand(Uniform(-1., 1.), cols)
        function df!(g, θ) # closure function for d(f_obj)/dθ
            g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
        end
        res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS(m=1_000), Optim.Options(show_trace=false, iterations=1_000))
        θ = Optim.minimizer(res)
        println(res)

        #= # linear solver:
        t = @elapsed begin
            θ = A\b
        end
        println("lin elapsed time: ", t)
        println("lin obj func = ", lsq(A, θ, b)) =#

        #r = residual(A, θ, b)
        #display(r)
        # ΔE:= |E_pred - E_actual|, independent of j (can pick any):
        MAE = 0.
        MADs = zeros(length(Widx)) # why use a list instead of just max? in case of multi MAD selection
        c = 1
        for m ∈ Widx
            # MAD_K(w), depends on j:
            MAD = 0.; VK = 0.
            for j ∈ Midx
                ΔjK, VK = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=true)
                MAD += abs(ΔjK)
            end
            MAD /= length(Midx)
            MADs[c] = MAD
            #println("MAD of m=$m is ", MAD)
            err = abs(VK - E[m])
            MAE += err
            c += 1       
        end
        MAE /= length(Widx)
        println("MAE of all mol w/ unknown E is ", MAE)
        # get the highest MAD:
        sidx = sortperm(MADs)[end]
        MADmax_idx = Widx[sidx]
        # get min |K| RMSD (the obj func):
        obj = Optim.minimum(res)
        
        println("largest MAD is = ", MADs[sidx], ", with index = ",MADmax_idx)
        # set a point with max MAD into the M:
        push!(Midx, MADmax_idx)
        filter!(!=(MADmax_idx), Widx)
        println([Midx, Widx])
        println("min K|∑RMSD(w) = ", obj)

        #= i = 1; j = Midx[i]; m = Widx[1]
        ΔjK = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=false)
        ΔjK_m = comp_ΔjK_m(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=false)
        display([r[i], A[i,:]'*θ - b[i], ΔjK, ΔjK_m]) # the vector slicing by default is column vector in Julia! =#
        
        # save MAE and 

        println()
    end
end



"""
test assemble A with dummy data
"""
function test_A()
    # data setup:
    n_data = 5; n_feature = 3; n_basis = 2
    bas = vec(1.:5.)
    W = zeros(n_feature, n_data)
    for i ∈ 1:n_feature
        W[i, :] = bas .+ (0.5*(i-1))
    end
    E = convert(Vector{Float64}, vec(1:5)) # dummy data matrix and energy vector
    display(W)
    D = convert(Matrix{Float64}, [0 1 2 3 4; 1 0 2 3 4; 1 2 0 3 4; 1 2 3 0 4; 1 2 3 4 0]) # dummy distance
    D = (D .+ D')./2
    display(D)

    Midx = [1,5] # k and j index
    data_idx = 1:n_data ; Widx = setdiff(data_idx, Midx) # unsupervised data index (m)
    cols = length(Midx)*n_feature*n_basis # index of k,l
    rows = length(Midx)*length(Widx) # index of j,m  
    bas = repeat([1.], n_feature)
    ϕ = zeros(n_feature, n_data, n_basis)
    for i ∈ 1:n_data
        for j ∈ 1:n_basis
            ϕ[:, i, j] = bas .+ 0.5*(j-1) .+ (i-1)
        end
    end
    # flattened basis*feature:
    ϕ = permutedims(ϕ, [1,3,2])
    ϕ = reshape(ϕ, n_feature*n_basis, n_data)
    #ϕ[1, :] .= 0.
    dϕ = ϕ*(-1.)
    display(ϕ)
    display(dϕ)

    A, b = assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis) # sparse ver
    println(A)
    println(b)
    # test each element:
    m = 2; j = 1; k = 1; l = 1
    ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
    SK = comp_SK(D, Midx, m)
    αj = SK*D[j,m] - 1; γk = SK*D[k,m]
    #println([ϕkl, SK, D[j,m], D[k,m], δ(j, k)])
    #println(ϕkl*(1-γk + δ(j, k)) / (γk*αj))

    # test predict V_K(w_m):
    θ = Vector{Float64}(1:cols) # dummy theta
    n_l =n_feature*n_basis
    ΔjK = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=true)
    display(ΔjK)

end
