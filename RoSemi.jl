using JLD

include("voronoi.jl")
include("linastic.jl")

"""
placeholder for the (Ro)bust (S)h(e)pard (m)odel for (i)nterpolation constructor
"""

function test()
    # load dataset || the datastructure is subject to change!! since the data size is huge
    dataset = load("data/ACSF_1000_symm.jld")["data"]
    N, L = size(dataset)
    A = dataset' # transpose data (becomes column major)
    display(A)
    println(N, " ", L)
    M = 100 # number of selected data
    # compute mean and cov:
    idx = 1 # the ith data point of the dataset
    wbar, C = mean_cov(A, idx, N, L)
    B = compute_B(C)
    display(wbar)
    display(B)
    # generate centers (M) for training:
    center_ids, mean_point = eldar_cluster(A, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd") # generate cluster centers
    display(mean_point)
    display(center_ids)
end