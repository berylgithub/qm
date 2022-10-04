using JLD, SparseArrays, Distributions, Statistics, StatsBase

include("voronoi.jl")
include("linastic.jl")

"""
placeholder for the (Ro)bust (S)h(e)pard (m)odel for (i)nterpolation constructor
"""


"""
query for
ϕ(w[m], w[k])[l] = ϕ(w[m])[l] - ϕ(w[k])[l] - ϕ'(w[k])[l]*(w[m] - w[k]) is the correct one; ϕ'(w)[l] = dϕ(w)[l]/dw
"""
function f_ϕ(ϕ, m, k, l)
    
end


"""
wrapper for scalar w for ϕ'(w) = dϕ(w)/dw
"""
function f_dϕ(x)
    return ForwardDiff.derivative(bspline_scalar, x)
end

"""
ϕ'(w) = dϕ(w)/dw using AD
params:
    - w, vector of features for a selected data, ∈ Float64 (n_feature) 
output:
    - y := ϕ'(w) ∈ Float64 (n_feature)
"""
function f_dϕ_vec(w)
    y = similar(w)
    for i ∈ eachindex(w)
        y[i] = ForwardDiff.derivative(bspline_scalar, w[i])
    end
    return y
end

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
mostly unused (although faster), verbose version
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
Bspline but assumes the input is a scalar, for efficient AD purpose
"""
function bspline_scalar(x)
    β = 0.
    z = abs(x)
    if z < 1
        β = 1 + .75*x^2 * (z - 2)
    elseif 1 ≤ z < 2
        β = .25*(2-z)^3
    end
    return β
end

"""
wrapper to extract M+3 or n_basis amount of splines
params:
    - x, matrix, ∈ Float64 (n_features, n_data) 
    - M, number of basfunc, returns M+3 basfunc
outputs:
    - S, array of basfuncs
        if flatten ∈ Float64 (n_feature*(M+3), n_data)
        else ∈ Float64 (n_feature, n_data, M+3)
"""
function extract_bspline(x, M; flatten=false)
    n_feature, n_data = size(x)
    n_basis = M+3
    S = zeros(n_feature, n_data, n_basis)
    for i ∈ 1:M+3
        S[:, :, i] = bspline(M .* x .+ 2 .- i) # should be M+3 features
    end
    if flatten # flatten the basis
        S = permutedims(S, [1,3,2])
        S = reshape(S, n_feature*n_basis, n_data)
    end
    return S
end

"""
extract both ϕ and dϕ
"""
function extract_bspline_df(x, M; flatten=false)
    n_feature, n_data = size(x)
    n_basis = M+3
    S = zeros(n_feature, n_data, n_basis)
    dϕ = zeros(n_feature, n_data, n_basis)
    @simd for i ∈ 1:n_basis
        @simd for j ∈ 1:n_data
            @simd for k ∈ 1:n_feature
                @inbounds S[k, j, i] = bspline_scalar(M*x[k, j] + 2 - i) # should be M+3 features
                @inbounds dϕ[k, j, i] = f_dϕ(M*x[k, j] + 2 - i)
            end
        end
    end
    if flatten # flatten the basis
        S = permutedims(S, [1,3,2])
        S = reshape(S, n_feature*n_basis, n_data)
        dϕ = permutedims(dϕ, [1,3,2])
        dϕ = reshape(dϕ, n_feature*n_basis, n_data)
    end
    return S, dϕ
end


"""
=============================
refer to RoSeMI.pdf and RSI.pdf for these quantities:
"""

"""
get the distance between w_l and w_k, D_k(w_l), uses precomputed matrix D with fixed i
params:
    - D, mahalanobis distance matrix, ∈ Float64 (n_data, n_data)
    - m, index of the selected unsupervised datapoint
    - k, index of the selected supervised datapoint
"""
function get_Dk(D, k, m)
    return D[k, m] # = D[l,k], same thing    
end

"""
compute S_K := ∑1/Dₖ
params:
    - D, mahalanobis distance matrix, ∈ Float64 (n_data, n_data)
    - m, index of the selected unsupervised datapoint
    - Midx, list of index of supervised datapoints, ∈ Vector{Int64}
"""
function comp_SK(D, Midx, m)
    sum = 0.
    for i ∈ Midx
        sum += 1/D[i, m]
    end
    return sum
end

function comp_γk(Dk, SK)
    return Dk*SK
end

function comp_αj(Dj, SK)
    return Dj*SK - 1
end


"""
==================================
"""


function test_spline()
    M = 5
    n_finger = 2
    n_data = Integer(100)
    x = [collect(LinRange(0., 1., 100)) collect(LinRange(0., 1., 100)) .+ 1]
    x = transpose(x)
    display(x)
    S = extract_bspline(x, M)
    for i ∈ 1:n_finger
        display(plot(vec(x[i,:]), S[i, :, :]))
    end
    # flattened feature*basis:
    #S = extract_bspline(x, M; flatten=true)

    # spline using scalar mode, see if the result is the same (and test with AD):
    S, dϕ = extract_bspline_df(x, M)
    for i ∈ 1:n_finger
        display(plot(vec(x[i,:]), S[i, :, :]))
        display(plot(vec(x[i,:]), dϕ[i, :, :]))
    end

end
