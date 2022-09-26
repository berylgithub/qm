using JLD, SparseArrays, Distributions, Statistics

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
    M = 5
    n_data = Integer(1e2)
    x = reshape(collect(LinRange(-1., 1., n_data)), 1, :) # data points with just 1 feature, matrix(1, ndata)
    S = zeros(n_data, M+3)
    for i ∈ 0:M
        S[:, i+2] = bspline(M.*x .- i) # should be M+3 features, but it seems the fist and last col is zeros
    end
    # set negatives to zeros:
    idx = S .< 0.
    S[idx] .= 0.
    display(x)
    display(S)
    plot(vec(x), S, xlims = (-1., 2.), ylims = (-.1, 1.))
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

function test_input_spline()
    # one dimensional normaliztion test:
    fhat = rand(Uniform(1., 10.), 10) # data vector size = (N)
    display(fhat)
    println("t = 1")
    supf = maximum(fhat)
    inff = minimum(fhat)
    b = (supf + inff)/2
    P = 1 / (supf - b)
    u = P*(fhat .- b)
    display(u)

    # dim t > 1:
    println("t > 1")
    fhat = rand(Uniform(1., 10.), (2, 10))
    display(fhat)
    supf = maximum(fhat, dims=2)
    inff = minimum(fhat, dims=2)
    b = (supf .+ inff) ./ 2
    display(b)
    P = diagm(vec(1 ./ (supf .- b)))
    u = P*(fhat .- b)
    display(u)


end