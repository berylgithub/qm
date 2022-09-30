using LsqFit, ReverseDiff, ForwardDiff, BenchmarkTools, Optim
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

NOTES:
    - for large LSes, ForwardDiff is much faster for jacobian but ReverseDiff is much faster for gradient !!
"""
function test_fit()
    #ndata = Int(1e4); nfeature=Int(1e4)
    ndata = 3; nfeature=3
    A = Matrix{Float64}(LinearAlgebra.I, 3,3)
    #A = rand(ndata, nfeature)
    #display(A)
    θ = rand(nfeature)
    b = ones(ndata) .+ 10.
    r = residual(A, θ, b)
    display(r)
    function df!(g, θ)
        g .= ReverseDiff.gradient(θ -> lsq(A, θ, b), θ)
    end
    res = optimize(θ -> lsq(A, θ, b), df!, θ, LBFGS())
    display(Optim.minimizer(res))
end

