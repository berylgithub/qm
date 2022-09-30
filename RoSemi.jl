using JLD, SparseArrays, Distributions, Statistics, StatsBase

include("voronoi.jl")
include("linastic.jl")

"""
placeholder for the (Ro)bust (S)h(e)pard (m)odel for (i)nterpolation constructor
"""


"""
The Bspline works for matrices

bspline constructor (mimic of matlab ver by prof. Neumaier)
params:
    - z, feature matrix (ndata, nfeature)
"""
function bspline(z)
    m, n = size(z)
    β = sparse(zeros(m, n))
    z = abs.(z)
    ind = (z .< 1)
    z1 = z[ind]
    β[ind] = 1 .+ 0.75*z1.^2 .*(z1 .- 2)
    ind = (.!ind) .&& (z .< 2)
    z1 = z[ind]
    β[ind] = 0.25*(2 .- z1).^3
    return β
end



"""
verbose version
"""
function bspline2(x)
    m, n = size(x) # fingerprint x data
    β = sparse(zeros(m, n))
    for j ∈ 1:n 
        for i ∈ 1:m
            z = abs(x[i,j])
            if z < 1
                β[i,j] = 1 + .75*x[i,j]^2 * (z - 2)
            elseif 1 ≤ z < 2
                β[i,j] = 0.25 * (2 - z)^3
            end
        end
    end
    return β
end

"""
wrapper to extract M+3 or n_basis amount of splines
params:
    - x, matrix, ∈ Float64 (n_features, n_data) 
    - M, number of basfunc, returns M+3 basfunc
outputs:
    - S, array of basfuncs, ∈ Float64 (n_feature, n_data, M+3)
"""
function extract_bspline(x, M)
    n_feature, n_data = size(x)
    S = zeros(n_feature, n_data, M+3)
    for i ∈ 1:M+3
        S[:, :, i] = bspline(M .* x .+ 2 .- i) # should be M+3 features, but it seems the fist and last col is zeros
    end
    return S
end


function test_spline()
    M = 5
    n_finger = 2
    n_data = Integer(100)
#=     x = reshape(collect(LinRange(0., 1., n_data)), 1, :) # data points with just 1 feature, matrix(1, ndata)
    S = zeros(n_data, M+3)
    for i ∈ 1:M+3
        S[:, i] = bspline2(M*x .+ 2 .- i) # should be M+3 features, but it seems the fist and last col is zeros
    end
    display(x)
    display(S)
    plot(vec(x), S)
    #, xlims = (-.5, 1.5), ylims = (-.1, 1.) =#
    x = [collect(LinRange(0., 1., 100)) collect(LinRange(0., 1., 100)) .+ 1]
    x = transpose(x)
    #= S = zeros(n_finger, n_data, M+3)
    for i ∈ 1:M+3
        S[:, :, i] = bspline(M .* x .+ 2 .- i) # should be M+3 features, but it seems the fist and last col is zeros
    end =#
    display(x)
    S = extract_bspline(x, M)
    display(S)
    for i ∈ 1:n_finger
        display(plot(vec(x[i,:]), S[i, :, :]))
    end
end

function test_cluster()
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
end


"""
compute all D_k(w_l) ∀k,l, for now fix i = 603 (rozemi)
"""
function test_all_dist()
    dataset = load("data/ACSF_1000_symm.jld")["data"]
    N, L = size(dataset)
    W = dataset' # transpose data (becomes column major)
    println(N, " ", L)
    M = 10 # number of selected data
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
test extract basis from data
"""
function test_basis_data()
    dataset = load("data/ACSF_1000_symm.jld")["data"]

end