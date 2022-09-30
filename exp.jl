"""
contains all tests and experiments
"""

include("voronoi.jl")
include("linastic.jl")
include("RoSemi.jl")



"""
compute the basis functions from normalized data
"""
function main_basis()
    W = load("data/ACSF_1000_symm_scaled.jld")["data"]'# load normalized fingerprint
    M = 10
    ϕ = extract_bspline(W, M) # compute basis from fingerprint
    #ϕ = sparse(ϕ[:,:,13]) # sparse can only take matrix
    
end