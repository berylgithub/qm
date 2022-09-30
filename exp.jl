using LsqFit
"""
contains all tests and experiments
"""

include("voronoi.jl")
include("linastic.jl")
include("RoSemi.jl")



"""
compute the basis functions from normalized data and assemble A matrix:
"""
function main_basis()
    W = load("data/ACSF_1000_symm_scaled.jld")["data"]'# load normalized fingerprint
    M = 10
    ϕ = extract_bspline(W, M) # compute basis from fingerprint ∈ (n_feature, n_data, M+3)
    #ϕ = sparse(ϕ[:,:,13]) # sparse can only take matrix
    # assemble matrix A:
end


"""
test for linear system fitting using leastsquares
"""
function test_lsfit()
    ndata = 3; nfeature=5
    A = rand(ndata, nfeature)
    θ = rand(nfeature)
    b = ones(ndata)
    curve_fit()
    
end

