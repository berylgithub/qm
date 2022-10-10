using JLD, SparseArrays, Distributions, Statistics, StatsBase, ForwardDiff

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
function extract_bspline_df(x, M; flatten=false, sparsemat=false)
    n_feature, n_data = size(x)
    n_basis = M+3
    if flatten # flatten the basis
        rsize = n_feature*n_basis
        S = zeros(rsize, n_data)
        dϕ = zeros(rsize, n_data)
        @simd for i ∈ 1:n_data
            rcount = 1
            @simd for j ∈ 1:n_basis
                @simd for k ∈ 1:n_feature
                    @inbounds S[rcount, i] = bspline_scalar(M*x[k, i] + 2 - j)
                    @inbounds dϕ[rcount, i] = f_dϕ(M*x[k, i] + 2 - j)
                    rcount += 1
                end
            end
        end
        if sparsemat # naive sparse, could still do smart sparse using triplets (I, J, V)
            S = sparse(S)
            dϕ = sparse(dϕ)
        end
    else # basis in last index of the array, possible for sparse matrix!!
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
    end
    return S, dϕ
end

"""
tis uses sparse logic with (I, J, V) triplets hence it will be much more efficient.
"""
function extract_bspline_sparse(x, M; flatten=false)
    n_feature, n_data = size(x)
    n_basis = M+3
    if flatten # flatten the basis
        rsize = n_feature*n_basis
        S = zeros(rsize, n_data)
        dϕ = zeros(rsize, n_data)
        @simd for i ∈ 1:n_data
            rcount = 1
            @simd for j ∈ 1:n_basis
                @simd for k ∈ 1:n_feature
                    @inbounds S[rcount, i] = bspline_scalar(M*x[k, i] + 2 - j)
                    @inbounds dϕ[rcount, i] = f_dϕ(M*x[k, i] + 2 - j)
                    rcount += 1
                end
            end
        end
    end
    #......
end



"""
query for
ϕ(w[m], w[k])[l] = ϕ(w[m])[l] - ϕ(w[k])[l] - ϕ'(w[k])[l]*(w[m] - w[k]) is the correct one; ϕ'(w)[l] = dϕ(w)[l]/dw,
currently uses the assumption of P = I hence P*w = w, the most correct one is β':= bspline'((P*w)_t), 
params:
    - l here corresponds to the feature index,
        used also to determine t, which is t = l % n_feature 
        *** NOT NEEDED ANYMORE ***if there exists B basis, hence there are M × l indices, i.e., do indexing of l for each b ∈ B, the indexing formula should be: |l|(b-1)+l, where |l| is the feature length
    - ϕ, basis matrix, ∈ Float64(n_s := n_feature*n_basis, n_data), arranged s.t. [f1b1, f2b1, ...., fnbn]
    - dϕ, the derivative of ϕ, idem to ϕ
    - W, feature matrix, ∈ Float64(n_feature, n_data)
    - m, index of selected unsup data
    - k, ... sup data 
"""
function qϕ(ϕ, dϕ, W, m, k, l, n_feature)
    t = l % n_feature # determine index of t, since dϕ is only non zero at t, hence the inner product is simplified
    if t == 0
        t = n_feature
    end
    #display([t, ϕ[l, m], ϕ[l, k], dϕ[l, k], W[t,m], W[t,k]])
    return ϕ[l,m] - ϕ[l, k] - dϕ[l, k]*(W[t,m]-W[t,k])
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
more "accurate" basis extractor to the formula: 
    ϕ_l(w) := β_τ((Pw)_t), l = (τ, t), where τ is the basis index, and t is the feature index, P is a scaler matrix (for now I with size of w)
"""
function β_τ(P, w)
    
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
assemble A matrix and b vector for the linear system, should use sparse logic (I, J, V triplets) later!!
params:
    - W, data × feature matrix, Float64 (n_feature, n_data)
    ...
"""
function assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    # assemble A (try using sparse logic later!!):
    n_m = length(Midx)
    n_w = length(Widx) # different from n_data!! n_data := size(W)[2]
    n_l = n_feature*n_basis
    rows = n_w*n_m
    cols = n_l*n_m
    A = zeros(rows, cols) 
    b = zeros(rows) 
    rcount = 1 #rowcount
    for m ∈ Widx
        SK = comp_SK(D, Midx, m)
        for j ∈ Midx
            ccount = 1 # colcount
            ∑k = 0. # for the 2nd term of b
            αj = SK*D[j,m] - 1
            for k ∈ Midx
                γk = SK*D[k, m]
                den = γk*αj
                ∑k = ∑k + E[k]/den # E_k/(γk × αj)
                for l ∈ 1:n_l # from flattened feature
                    ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
                    #display(ϕkl)
                    num = ϕkl*(1-γk + δ(j, k)) # see RoSemi.pdf and RSI.pdf for ϕ and dϕ definition
                    A[rcount, ccount] = num/den
                    ccount += 1 # end of column loop
                end
            end
            b[rcount] = E[j]/αj - ∑k # assign b vector elements
            rcount += 1 # end of row loop
        end
    end
    return A, b
end

"""
assemble A matrix and b vector for the linear system, with sparse logic (I, J, V triplets)
params:
    ...
"""
function assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    
end


"""
predict the energy of w_m by computing V_K(w_m), naive or fair(?) version, since all quantities except phi are recomputed
params:
    - W, fingerprint matrix, ∈Float64(n_feature, n_data)
    - ...
    - m, index of W in which we want to predict the energy
    - n_l := n_basis*n_feature, length of the feature block vector,
output:
    - VK, scalar Float64
notes:
    - for m = k, this returns undefined or NaN by definition of V_K(w).
"""
function comp_VK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m)
    SK = comp_SK(D, Midx, m) # compute SK
    RK = 0.
    ccount = 1 # the col vector count, should follow k*l, easier to track than trying to compute the indexing pattern.
    for k ∈ Midx
        ∑l = 0. # right term with l index
        for l ∈ 1:n_l # ∑θ_kl*ϕ_kl
            ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
            #kl_idx = n_l*(k-1) + l # use the indexing pattern # doesnt work if the index is contiguous
            θkl = θ[ccount] #θ[kl_idx]  # since θ is in block vector of [k,l]
            ∑l = ∑l + θkl*ϕkl
            #println([ccount, θkl, ϕkl, ∑l])
            ccount += 1
        end
        vk = E[k] + ∑l
        RK = RK + vk/D[k, m] # D is symm
        #println([E[k], ∑l, D[k, m], RK])
    end
    #println(SK)
    return RK/SK
end

"""
compute Δ_jK(w_m). Used for MAD and RMSD. See comp_VK function, since Δ_jK(w_m) := (VK - Vj)/αj
"""
function comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk = false)
    SK = comp_SK(D, Midx, m) # compute SK
    RK = 0.
    ∑l_j = 0. # for j indexer, only passed once and j ∈ K
    ccount = 1 # the col vector count, should follow k*l, easier to track than trying to compute the indexing pattern.
    for k ∈ Midx
        ∑l = 0. # right term with l index
        for l ∈ 1:n_l # ∑θ_kl*ϕ_kl
            # for k:
            ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
            θkl = θ[ccount] # since θ is in block vector of [k,l]
            θϕ = θkl*ϕkl
            ∑l = ∑l + θϕ
            if k == j # for j terms:
                ∑l_j = ∑l_j + θϕ
            end
            #println([ccount, θkl, ϕkl, ∑l, ∑l_j])
            ccount += 1
        end
        vk = E[k] + ∑l
        RK = RK + vk/D[k, m] # D is symm
        #println([E[k], ∑l, D[k, m], RK])
    end
    #println(SK)
    VK = RK/SK
    αj = D[j, m]*SK - 1
    Vj = E[j] + ∑l_j
    if return_vk
        return (VK - Vj)/αj, VK
    else
        return (VK - Vj)/αj
    end
end

"""
MAD_k(w_m) := 1/|K| ∑_j∈K |ΔjK(w_m)| 
"""
function MAD(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m)
    len = length(Midx)
    ∑ = 0.
    for j ∈ Midx
        ∑ = ∑ + abs(comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j))
    end
    return ∑/len
end

"""
only from a list of ΔjK
"""
function MAD(ΔjKs)
    len = length(ΔjKs)
    return sum(abs.(ΔjKs))/len
end

"""
MAE := ∑i|ΔE_i|
"""

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
    display(dϕ)
    for i ∈ 1:n_finger
        display(plot(vec(x[i,:]), S[i, :, :]))
        display(plot(vec(x[i,:]), dϕ[i, :, :]))
    end
    S, dϕ = extract_bspline_df(x, M; flatten = true, sparsemat=true)
    display(dϕ)
    #S = sparse(S); dϕ = sparse(dϕ)
    display([nnz(S), nnz(dϕ)])
end
