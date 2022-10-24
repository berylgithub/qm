using JLD, SparseArrays, Distributions, Statistics, StatsBase, ForwardDiff, ReverseDiff

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
    t = l % n_feature # find index t given index l and length of feature vector chosen (or n_f = L/n_b)
    if t == 0
        t = n_feature
    end
    return ϕ[l,m] - ϕ[l, k] - dϕ[l, k]*(W[t,m]-W[t,k]) # ϕ_{kl}(w_m) := ϕ_l(w_m) - ϕ_l(w_k) - ϕ_l'(w_k)(w_m - w_k), for k ∈ K, l = 1,...,L 
end


"""
for (pre-)computing ϕ_{kl}(w_m) := ϕ_l(w_m) - ϕ_l(w_k) - ϕ_l'(w_k)(w_m - w_k), for k ∈ K (or k = 1,...,M), l = 1,...,L, m ∈ Widx, 
compute (and sparsify outside) B := B_{m,kl}, this is NOT a contiguous matrix hence it is indexed by row and column counter
instead of directly m and kl.
params mostly same as qϕ
"""
function comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature)
    klc = 1                                                     # kl counter
    for k ∈ Midx
        for l ∈ 1:L
            rc = 1                                              # the row entry is not contiguous
            for m ∈ Widx
                B[rc, klc] = qϕ(ϕ, dϕ, W, m, k, l, n_feature) 
                rc += 1
            end
            klc += 1
        end
    end
end

"""
kl index computer, which indexes the column of B
params:
    - M, number of sup data
    - L, n_feature*n_basis
"""
function kl_indexer(M, L)
    klidx = Vector{UnitRange}(undef, M) # this is correct, this is the kl indexer!!
    c = 1:M
    for i ∈ c
        n = (i-1)*L + 1 
        klidx[i] = n:n+L-1
    end
    return klidx
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
returns a matrix with size m × j
params:
    SKs, precomputed SK vector ∈ Float64(N)
"""
function comp_α(D, SKs, Midx, Widx)
    J = length(Midx); N = length(Widx)
    α = zeros(N, J)
    mc = 1
    for m ∈ Widx
        jc = 1
        for j ∈ Midx
            α[mc, jc] = D[j, m]*SKs[mc] - 1
            jc += 1
        end
        mc += 1
    end
    return α
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
                    num = ϕkl*(1-γk*δ(j, k)) # see RoSemi.pdf and RSI.pdf for ϕ and dϕ definition
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
assemble A matrix and b vector for the linear system, with sparse logic (I, J, V triplets), dynamic vectors at first, 3x slower than static ones though
params:
    ...
"""
function assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    n_m = length(Midx)
    n_w = length(Widx) # different from n_data!! n_data := size(W)[2]
    n_l = n_feature*n_basis
    rows = n_w*n_m
    cols = n_l*n_m
    b = zeros(rows)
    J = Vector{Int64}(undef, 0); K = Vector{Int64}(undef, 0); V = Vector{Float64}(undef, 0); # empty vectors
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
                    num = ϕkl*(1-γk*δ(j, k)) # see RoSemi.pdf and RSI.pdf for ϕ and dϕ definition
                    val = num/den
                    # assign the vectors:
                    if val != 0. # check if it's nonzero then push everything
                        push!(J, rcount)
                        push!(K, ccount)
                        push!(V, val)
                    else
                        if (rcount == rows) && (ccount == cols) # final entry trick, push zeros regardless
                            push!(J, rcount)
                            push!(K, ccount)
                            push!(V, val)
                        end
                    end
                    ccount += 1 # end of column loop
                end
            end
            b[rcount] = E[j]/αj - ∑k # assign b vector elements
            rcount += 1 # end of row loop
        end
    end
    A = sparse(J, K, V)
    return A, b
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
    SK = comp_SK(D, Midx, m) # SK(w_m)
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
        RK = RK + vk/D[k, m]
        #println([E[k], ∑l, D[k, m], RK])
    end
    #println(SK)
    VK = RK/SK
    αj = D[j, m]*SK - 1.
    Vj = E[j] + ∑l_j
    #println([VK, Vj, αj])
    if return_vk
        return (VK - Vj)/αj, VK
    else
        return (VK - Vj)/αj
    end
end


"""
overloader for ΔjK, use precomputed distance matrix D and SK[m] 
"""
function comp_v_jm(W, E, D, θ, ϕ, dϕ, SK, Midx, n_l, n_feature, m, j)
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
        RK = RK + vk/D[k, m]
        #println([E[k], ∑l, D[k, m], RK])
    end
    VK = RK/SK
    αj = D[j, m]*SK - 1.
    Vj = E[j] + ∑l_j
    #println([VK, Vj, αj])
    return (VK - Vj)/αj
end


"""
computes the A*x := ∑_{kl} θ_kl ϕ_kl (1 - γ_k δ_jk)/γ_k α_j
Same as v_j function but for VK only
"""
function comp_Ax!(Ax, b, temps, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, γ, α)
    vk, RK, VK, ϕkl, ϕjl = temps;
end

"""
computes b := (E_j - ∑_k E_k/γ_k α_j)
"""
function comp_b!()
    
end

"""
computes v_j := ΔjK ∀m (returns a vector with length m), with precomputed vector of matrices B instead of (W, ϕ, dϕ)
params:
    - klidx, precomputed θ indexer, since θ is a block vector

output:
    - v_j, a vector of length N (or n_unsup_data)
"""
function comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, αj, j)
    ΔjK, vk, vj, RK, VK, ϕkl, ϕjl = outs; # move these outside later, to avoid alloc
    @simd for c ∈ cidx # vectorized op on N vector length s.t. x = [m1, m2, ... N]
        k = Midx[c]
        ϕkl .= B[:,klidx[c]]*θ[klidx[c]]
        @. vk = E[k] + ϕkl
        @. RK = RK + (vk/D[k, Widx])
        if j == k # for j terms
            ϕjl .= ϕkl
        end
    end
    @. VK = RK / SKs
    @. vj = E[j] + ϕjl
    @. ΔjK = (VK - vj) / αj
end

"""
full ΔjK computer ∀jm, m × j matrix
"""
function comp_v!(v, outs, temp, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α)
    for jc ∈ cidx
        comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α[:, jc], Midx[jc])
        v[:, jc] .= outs[1]
        outs .= temp
    end
end


"""
compute the whole vector v with components v_jm := ΔjK(w_m)
output:
    - v, vector with length N × M
"""
function comp_v!(v, W, E, D, θ, ϕ, dϕ, SKs, Widx, Midx, n_l, n_feature)
    c = 1
    skc = 1
    @simd for m ∈ Widx
        @simd for j ∈ Midx 
           @inbounds v[c] = comp_v_jm(W, E, D, θ, ϕ, dϕ, SKs[skc], Midx, n_l, n_feature, m, j)
           c += 1
        end
        skc += 1
    end
end


"""
for AD, since comp_v_j is vectorized
"""
function comp_v_jm()
    
end

"""
UNUSED
compute Δ_jK in terms of matrix vector mult, see if this checks out
"""
function comp_ΔjK_m(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk = false)
    SK = comp_SK(D, Midx, m)
    αj = D[j, m]*SK - 1.
    ccount = 1
    ∑kl = 0. # left
    ∑kr = 0. # right
    for k ∈ Midx
        γk = D[k, m]*SK
        den = γk*αj
        rt = E[k]/den # right term, depends on k only
        ∑kr += rt
        ∑l = 0.
        for l ∈ 1:n_l            
            # left term:
            num = θ[ccount]*qϕ(ϕ, dϕ, W, m, k, l, n_feature)*(1 - γk*δ(j,k))
            ∑l += num
            ccount += 1
        end
        ∑kl = ∑kl + ∑l/den
    end
    b = E[j]/αj - ∑kr
    A = ∑kl
    return A-b
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
    println("W = ")
    display(W)
    D = convert(Matrix{Float64}, [0 1 2 3 4; 1 0 2 3 4; 1 2 0 3 4; 1 2 3 0 4; 1 2 3 4 0]) # dummy distance
    D = (D .+ D')./2
    println("D = ")
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
    #ϕ[1, :] .= 0.
    dϕ = ϕ*(-1.)
    println("ϕ = ")
    display(ϕ)
    println("dϕ = ")
    display(dϕ)

    A, b = assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis) # sparse ver
    display(A)
    println(b)
    # test each element:
    #m = Widx[1]; j = Midx[1]; k = Midx[1]; l = 1
    #ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
    #αj = SK*D[j,m] - 1; γk = SK*D[k,m]
    #println([ϕkl, SK, D[j,m], D[k,m], δ(j, k)])
    #println(ϕkl*(1-γk + δ(j, k)) / (γk*αj))

    # tests for vjm vs Aθ-b:
    #= SKs = map(m -> comp_SK(D, Midx, m), Widx) # precompute vector of SK ∈ R^N for each set of K
    display(SKs)
    # test predict V_K(w_m):
    θ = Vector{Float64}(1:cols) # dummy theta
    n_l =n_feature*n_basis
    ΔjK = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk=false)
    
    v_jm = zeros(length(Widx)*length(Midx))
    c = 1
    skc = 1
    for m ∈ Widx
        SK = SKs[skc]
        skc += 1
        for j ∈ Midx
            v_jm[c] = comp_v_jm(W, E, D, θ, ϕ, dϕ, SK, Midx, n_l, n_feature, m, j)        
            c += 1
        end
    end
    v = zeros(length(Widx)*length(Midx))
    comp_v!(v, W, E, D, θ, ϕ, dϕ, SKs, Widx, Midx, n_l, n_feature)
    display([v_jm v (A*θ - b)]) #
    SK = comp_SK(D, Midx, m)
    display(ReverseDiff.gradient(θ -> comp_v_jm(W, E, D, θ, ϕ, dϕ, SK, Midx, n_l, n_feature, m, j), θ))
    display(ReverseDiff.jacobian(θ -> A*θ - b, θ)) =#

    # tests for precomputing the ϕkl:
    θ = Vector{Float64}(1:cols) # dummy theta
    println("W = ")
    display(W)
    println("ϕ = ")
    display(ϕ)
    println("dϕ = ")
    display(dϕ)
    M = length(Midx); N = length(Widx); L = n_feature*n_basis
    mc = 2; jc = 2; kc = 1
    m = Widx[mc]; j = Midx[jc]; k = Midx[kc]; l = 1
    SK = comp_SK(D, Midx, m)
    SKs = map(m -> comp_SK(D, Midx, m), Widx) # precompute vector of SK ∈ R^N for each set of K
    B = zeros(N, M*L)
    comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature) # the index should be k,l only
    display(B)
    klidx = kl_indexer(M, L) # this is correct, this is the kl indexer!!
    k = 2
    display(klidx)
    display(B[:,klidx[k]])
    display(θ[klidx[k]])
    display(B[:,klidx[k]]*θ[klidx[k]])
    α = comp_α(D, SKs, Midx, Widx) # precompute alpha matrix for each jm
    outs = [zeros(N) for _ = 1:7];
    cidx = 1:M # k indexer
    comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α[:,jc], j) # this is the tested one
    ΔjK = outs[1]
    ΔjK_act, VK_act = comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, L, n_feature, m, j; return_vk = true) # this is the correct one
    println([ΔjK, ΔjK_act])
    v = zeros(N, M)
    outs = [zeros(N) for _ = 1:7]; temp = [zeros(N) for _ = 1:7]
    comp_v!(v, outs, temp, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α) 
    display(v)
    #ReverseDiff.jacobian(θ->comp_v_j(E, D, θ, B, SKs, Midx, Widx, klidx, j), θ) # for AD, use each jm index and loop it instead of taking the jacobian (slow)
end

"""
test the timing of v vs Aθ - b
"""
function testtime()
    # setup data:
    n_data = 250; n_feature = 40; n_basis = 8
    W = rand(n_feature, n_data)
    E = rand(n_data)
    # setup data selection:
    M = 100; N = n_data - M
    dataidx = 1:n_data
    Midx = sample(dataidx, M, replace=false)
    Widx = setdiff(dataidx, Midx)
    # compute D, S and ϕ:
    t_data = @elapsed begin
        #= D = rand(n_data, n_data)
        D = (D + D')/2
        D[diagind(D)] .= 0. =#
        Bhat = Matrix{Float64}(I, n_feature, n_feature)
        D = compute_distance_all(W, Bhat)
        SKs = map(m -> comp_SK(D, Midx, m), Widx)
        α = comp_α(D, SKs, Midx, Widx)
        ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true)
        n_basis += 3; L = n_feature*n_basis # reset L
        B = zeros(N, M*L); comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature);
    end
    θ = rand(L*M)
    # assemble systems and compare!!:
    t_as = @elapsed begin
        # assemble A and b:
        A, b = assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis) #A, b = assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    end
    t_ls = @elapsed begin
        r_ls = A*θ - b
    end 
    v = zeros(N, M) #v = zeros(M,N)
    outs = [zeros(N) for _ = 1:7]; temp = [zeros(N) for _ = 1:7]; # temporary vars
    klidx = kl_indexer(M, L); cidx = 1:M # indexers
    t_v = @elapsed begin
        comp_v!(v, outs, temp, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α) #comp_v!(v, W, E, D, θ, ϕ, dϕ, SKs, Widx, Midx, n_l, n_feature)    
    end
    mems = [Base.summarysize(A), Base.summarysize(b), Base.summarysize(D), Base.summarysize(SKs), Base.summarysize(ϕ), Base.summarysize(dϕ)].*1e-6 # get storages
    println(mems)
    println([t_data, t_as, t_ls, t_v])
    println("norm(v - (Aθ-b)) = ",norm(r_ls - vec(v)))
    println("M = $M, N = $n_data, L = $n_feature × $n_basis = $L")
    println("ratio of mem(A)+mem(b)/(mem(D)+mem(S)+mem(ϕ)+mem(dϕ)) = ", sum(mems[1:2])/sum(mems[3:end]))
    println("ratio of time(A)+time(b)/(time(D)+time(S)+time(ϕ)+time(dϕ)) = ", t_as/t_data)
    println("ratio of time(Ax-b given A and b)/time(v given D, S, ϕ, and dϕ) = ", t_ls/t_v)
    
end

"""
tests LS without forming A (try Krylov.jl and Optim.jl)
"""
function test_LS()
    # try arbitrary system:
    row = 3; col = 5
    b = Vector{Float64}(1:row)
    A = rand(row, col)
    function Ax!(y, A, u)
        y .= A*u
    end
    function Aᵀb!(y, A, v)
        y .= A'*v
    end
    op = LinearOperator(Float64, row, col, false, false,    (y,u) -> Ax!(y,A,u),
                                                            (y,v) -> Aᵀb!(y,A,v))
    x, stat = cgls(op, b)
    display([A*x b])
    display(norm(A*x - b))
end