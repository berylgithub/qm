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
    Widx = Widx[1:100] # take subset for smaller matrix
    #display(dataset)
    n_m = length(Midx); n_w = length(Widx)
    display([length(data_idx), n_m, n_w])
    
    ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true) # compute basis from fingerprint ∈ (n_feature*(n_basis+3), n_data)
    n_basis += 3 # by definition of bspline
    display(ϕ)
    #println(maximum(ϕ), minimum(ϕ))
    #display([nnz(ϕ), nnz(dϕ)]) # only ≈1/3 of total entry is nnz
    #display(Base.summarysize(ϕ)) # turns out only 6.5mb for sparse
    display([n_data, n_basis, n_feature])
    # assemble A and b:
    t = @elapsed begin
        A, b = assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    end
    println("LS assembly time: ",t)
    A = sparse(A) # only half is filled!!
    display(A)
    #display(b)

    # fit, try lsovle vs lsquares!:
    n_l = n_basis*n_feature # length of feature*basis each k
    cols = n_m*n_l # length of col
    θ = rand(Uniform(-1., 1.), cols)
    r = residual(A, θ, b)
    display(r)
    function df!(g, θ) # closure function for d(f_obj)/dθ
        g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
    end
    res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS(m=1_0), Optim.Options(show_trace=true, iterations=1_00))
    θ_lsq = Optim.minimizer(res)
    display(res)
    # linear solver:
    #= t = @elapsed begin
        θ_lin = A\b
    end
    println("lin elapsed time: ", t)
    println("lin obj func = ", lsq(A, θ_lin, b)) =#
    display(residual(A, Optim.minimizer(res), b))
    println("ls obj func = ", lsq(A, θ_lsq, b))
    #println("differences of lin and LFBGS? ", norm(θ_lsq-θ_lin))

    # ΔE:= |E_pred - E_actual|:
    println("'test' acc:")
    MAE = 0.
    for m ∈ Widx
        E_actual = E[m] # actual
        VK = comp_VK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m) # predicted
        err = abs(VK - E_actual)
        MAE += err
        println([E_actual, VK])
        println("m = ",m,", ΔE = ",err)
    end
    MAE /= length(Widx)
    println(MAE)

    println("'train' acc:")
    MAE = 0.
    for j ∈ Midx
        E_actual = E[j] # actual
        VK = comp_VK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, j) # predicted
        err = abs(VK - E_actual)
        MAE += err
        println([E_actual, VK])
        println("j = ",j,", ΔE = ",err)
    end
    MAE /= length(Midx)
    println(MAE)
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

    A, b = assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis) # assemble the matrix A and vector b!!
    display(A)
    println(b)
    
    # test each element:
    m = 2; j = 5; k = 1; l = 1
    ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
    SK = comp_SK(D, Midx, m)
    αj = SK*D[j,m] - 1; γk = SK*D[k,m]
    println([ϕkl, SK, D[j,m], D[k,m], δ(j, k)])
    println(ϕkl*(1-γk + δ(j, k)) / (γk*αj))

    # test predict V_K(w_m):
    θ = Vector{Float64}(1:cols) # dummy theta
    display(θ)
    n_l =n_feature*n_basis
    VK = comp_VK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m)
    display(VK)
    ΔjK = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=true)
    display(ΔjK)
    MAD_m = MAD(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m)
    display(MAD_m)
    # test fitting !! (although the data is nonsensical (dummy))
#=     θ = rand(cols)
    r = residual(A, θ, b)
    println("residual = ", r)
    function df!(g, θ)
        g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
    end
    res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS())
    display(Optim.minimizer(res))
    display(res) =#
end

function spassign(X)
    r = c = length(X)
    K = Vector{Float64}(undef, 0); J = Vector{Float64}(undef, 0); V = Vector{Float64}(undef, 0) # assume unknown number of data
    for k ∈ 1:c 
        for j ∈ 1:r 
            if k == j
                push!(K, k); push!(J, j); push!(V, X[k])
            end
        end
    end
    return sparse(K, J, V)
end

function densassign(X)
    r = c = length(X)
    A = zeros(r, c)
    for i ∈ 1:c
        for j ∈ 1:r
            if j == i
                A[j, i] = X[j]
            end
        end
    end
    return sparse(A)
end

"""
test sparse vs dense loop, assume diagonal matrix
"""
function test_sparse()
    
    
end