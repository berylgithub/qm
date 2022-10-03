using LsqFit, ReverseDiff, ForwardDiff, BenchmarkTools, Optim
"""
contains all tests and experiments
"""

include("voronoi.jl")
include("linastic.jl")
include("RoSemi.jl")


"""
get the indices of the supervised datapoints M, fix w0 as i=603 for now
"""
function get_cluster()
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
function get_all_dist()
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
test for linear system fitting using leastsquares

NOTES:
    - for large LSes, ForwardDiff is much faster for jacobian but ReverseDiff is much faster for gradient !!
"""
function test_fit()
    #ndata = Int(1e4); nfeature=Int(1e4)
    ndata = 3; nfeature=3
    #A = Matrix{Float64}(LinearAlgebra.I, 3,3)
    #A = rand(ndata, nfeature)
    A = spzeros(ndata, nfeature) # try sparse
    for i ∈ 1:ndata
        for j ∈ 1:nfeature
            if j == i
                A[j,i] = 1.
            end
        end
    end
    display(A)
    θ = rand(nfeature)
    b = ones(ndata) .+ 10.
    r = residual(A, θ, b)
    display(r)
    function df!(g, θ)
        g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
    end
    res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS())
    display(Optim.minimizer(res))
    display(res)
end


"""
compute the basis functions from normalized data and assemble A matrix:
"""
function get_A()
    W = load("data/ACSF_1000_symm_scaled.jld")["data"]' # load and transpose the normalized fingerprint
    D = load("data/distances_1000_i=603.jld")["data"] # the mahalanobis distance matrix
    list_M = load("data/M=10_idx_1000.jld")["data"] # the supervised data points' indices

    n_basis = 10
    ϕ = extract_bspline(W, n_basis) # compute basis from fingerprint ∈ (n_feature, n_data, n_basis+3)
    #display(sizeof(ϕ)) # turns out only 10mb
    # determine size of A:
    s_W = size(W) # n_feature x n_data
    s_M = size(list_M) # n_sup_data
    s_ϕ = size(ϕ) # n_feature x n_data x n_basis+3
    N = s_W[2] # number of data (total)
    M = s_M[1] # number of centers (supervised data)
    L = s_ϕ[1]*s_ϕ[3] # length of feature
    display([N, M, L])
    row_size = N*M
    col_size = M*L
    A = spzeros(row_size, col_size) # init A
    #ϕ = sparse(ϕ[:,:,13]) # sparse can only take matrix
    # assemble matrix A:
end




