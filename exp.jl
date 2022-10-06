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
compute the basis functions from normalized data and assemble A matrix:
"""
function fit_rosemi()
    dataset = load("data/qm9_dataset_1000.jld") # energy is from here
    W = load("data/ACSF_1000_symm_scaled.jld")["data"]' # load and transpose the normalized fingerprint
    D = load("data/distances_1000_i=603.jld")["data"] # the mahalanobis distance matrix
    list_M = load("data/M=10_idx_1000.jld")["data"] # the supervised data points' indices

    n_basis = 10
    ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true) # compute basis from fingerprint ∈ (n_feature, n_data, n_basis+3)
    display(ϕ)
    display([nnz(ϕ), nnz(dϕ)])
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
    # naive and slow: assemble matrix A's entries: # loop m index first then j index for row, l first then k for col:
    for j ∈ 1:col_size 
        for i ∈ 1:row_size
            break
            #A[i,j] = 
        end
    end

end



"""
AD test for gradient vs Jacobian for a vector argument, it appears using ForwardDiff.derivative of a function which accepts scalar is multitude faster
"""
function test_AD()
    f(x) = x.^2
    ff(x) = x^2
    n = Int(1e4)
    x = rand(n)
    # jacobian:
    ReverseDiff.jacobian(f, x)
    # loop of gradient:
    y = similar(x)
    function df!(y, x)
        for i ∈ eachindex(x)
            y[i] = ForwardDiff.derivative(ff, x[i])
        end
    end
    df!(y, x)
end

"""
test for linear system fitting using leastsquares

NOTES:
    - for large LSes, ForwardDiff is much faster for jacobian but ReverseDiff is much faster for gradient !!
"""
function test_fit()
    ndata = Int(1e4); nfeature=Int(1e3)
    #ndata = 100; nfeature=500
    #A = Matrix{Float64}(LinearAlgebra.I, 3,3)
    A = rand(ndata, nfeature)
    #= A = spzeros(ndata, nfeature) # try sparse
    for i ∈ 1:ndata
        for j ∈ 1:nfeature
            if j == i
                A[j,i] = 1.
            end
        end
    end =#
    display(A)
    θ = rand(nfeature)
    #b = ones(ndata) .+ 10.
    b = rand(ndata)
    r = residual(A, θ, b)
    display(r)
    function df!(g, θ)
        g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
    end
    res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS(), Optim.Options(show_trace=true))
    display(Optim.minimizer(res))
    display(res)
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
    dϕ = ϕ*(-1.)
    display(ϕ)
    display(dϕ)

    A, b = gen_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    println(A)
    println(b)
    
    # test each element:
    m = 2; j = 1; k = 1; l = 1
    ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
    SK = comp_SK(D, Midx, m)
    αj = SK*D[j,m] - 1; γk = SK*D[k,m]
    println([ϕkl, SK, D[j,m], D[k,m], δ(j, k)])
    println(ϕkl*(1-γk + δ(j, k)) / (γk*αj))

    # test fitting !! (although the data is nonsensical (dummy))
    θ = rand(cols)
    r = residual(A, θ, b)
    println("residual = ", r)
    function df!(g, θ)
        g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
    end
    res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS())
    display(Optim.minimizer(res))
    display(res)
end
