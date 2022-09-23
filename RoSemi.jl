using JLD, SparseArrays

include("voronoi.jl")
include("linastic.jl")

"""
placeholder for the (Ro)bust (S)h(e)pard (m)odel for (i)nterpolation constructor
"""


"""
bspline constructor (mimic of matlab ver by prof. Neumaier)
params:
    - x, feature matrix (ndata x nfeature)
"""
function bspline(x)
    m, n = size(x)
    s = sparse(zeros(m, n))
    x = abs.(x)
    ind = x .< 1
    x1 = x[ind]
    s[ind] = 1 .+ .75*x1.^2 .*(x1 .- 2)
    ind = (s .< 2) .&& (.~ind)
    x1 = x[ind]
    s[ind] = .25*(2 .- x1).^3
    return s
end

function test_spline()
    M = 6
    n_data = Integer(1e3)
    x = reshape(collect(LinRange(-1., 1., n_data)), 1, :) # data points with just 1 feature, matrix(1, ndata)
    S = Matrix{Float64}(undef, n_data, M+1)
    for i ∈ 0:M
        S[:, i+1] = bspline(M.*x .- i)
    end
    display(x)
    display(S)
    plot(vec(x), S, xlims = (-1., 2.), ylims = (-1., 1.))
end

function test_cluster()
    # load dataset || the datastructure is subject to change!! since the data size is huge
    dataset = load("data/ACSF_1000_symm.jld")["data"]
    N, L = size(dataset)
    A = dataset' # transpose data (becomes column major)
    display(A)
    println(N, " ", L)
    M = 100 # number of selected data
    # compute mean and cov:
    idx = 200 # the ith data point of the dataset, can be arbitrary technically
    wbar, C = mean_cov(A, idx, N, L)
    B = compute_B(C)
    display(wbar)
    display(B)
    # generate centers (M) for training:
    center_ids, mean_point = eldar_cluster(A, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd") # generate cluster centers
    display(mean_point)
    display(center_ids)
end