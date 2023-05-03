using JLD, SparseArrays, Distributions, Statistics, StatsBase, ForwardDiff, ReverseDiff, LinearOperators, Krylov
using .Threads

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
compute S_K := ∑1/Dₖ
params:
    - D, mahalanobis distance matrix, ∈ Float64 (n_data, n_data)
    - m, index of the selected unsupervised datapoint
    - Midx, list of index of supervised datapoints, ∈ Vector{Int64}
"""
function comp_SK(D, Midx, m)
    sum = 0.
    for i ∈ eachindex(Midx)
        sum += 1/D[m, i] # not the best indexing way...
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
function comp_γ(D, SKs, Midx, Widx)
    M = length(Midx); N = length(Widx)
    γ = zeros(N, M)
    for kc ∈ eachindex(Midx)
        for mc ∈ eachindex(Widx)
            m = Widx[mc]
            γ[mc, kc] = D[m, kc]*SKs[mc]
        end
    end
    return γ
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
function comp_Ax_j!(temps, θ, B, Midx, cidx, klidx, γ, α, j, jc)
    ∑k, num, den = temps;
    @simd for c ∈ cidx  # vectorized op on N vector length s.t. x = [m1, m2, ... N]
        k = Midx[c]
        num .= (@view B[:,klidx[c]])*θ[klidx[c]] .* (1 .- (@view γ[:,c]).*δ(j,k))
        den .= (@view γ[:,c]) .* (@view α[:, jc])
        ∑k .+= (num ./ den)
        #∑k .+= /( (@view γ[:,k]) .* (@view α[:, j])) # of length N
    end
end

function comp_Ax!(Ax, Axtemp, temps, θ, B, Midx, cidx, klidx, γ, α)
    # loop for all j:
    for jc ∈ cidx
        comp_Ax_j!(temps, θ, B, Midx, cidx, klidx, γ, α, Midx[jc], jc)
        Axtemp[:, jc] .= temps[1]
        fill!.(temps, 0.)
    end
    Ax .= vec(transpose(Axtemp)) # transpose first for j as inner index then m outer
end

"""
computes b_j := (E_j - ∑_k E_k/γ_k α_j) ∀m for each j, 
"""
function comp_b_j!(temps, E, γ, α, Midx, cidx, j, jc)
    b_j, ∑k = temps;
    for c ∈ cidx
        k = Midx[c]
        @. ∑k = ∑k + (E[k] / (@view γ[:, c])) # ∑_k E_k/γ_k(w_m) , E has absolute index (j, k) while the others are relative indices (jc, mc)
    end
    @. b_j = (E[j] - ∑k) / (@view α[:, jc])
end

function comp_b!(b, btemp, temps, E, γ, α, Midx, cidx)
    for jc ∈ cidx
        comp_b_j!(temps, E, γ, α, Midx, cidx, Midx[jc], jc)
        btemp[:, jc] .= temps[1]
        fill!.(temps, 0.)
    end
    b .= vec(transpose(btemp))  
end

"""
computes Aᵀv, where v ∈ Float64(col of A), required for CGLS
params:
    
"""
function comp_Aᵀv!(Aᵀv, v, B, Midx, Widx, γ, α, L)
    rc = 1 # row counter
    for kc ∈ eachindex(Midx)
        k = Midx[kc]; # absolute index k
        for l ∈ 1:L
            cc = 1 # col counter
            ∑ = 0.
            for mc ∈ eachindex(Widx)
                for jc ∈ eachindex(Midx)
                    j = Midx[jc]; # absolute index j
                    num = B[mc, rc]*v[cc]*(1 - γ[mc,kc]*δ(j,k))
                    den = γ[mc, kc]*α[mc, jc]
                    ∑ = ∑ + num/den
                    cc += 1
                end
            end
            Aᵀv[rc] = ∑
            rc += 1
        end
    end
end

"""
computes ΔjK := ΔjK for m = 1,...,N (returns a vector with length N), with precomputed vector of matrices B instead of (W, ϕ, dϕ)
params:
    - outs, temporary vectors to avoid memalloc
    - E, energy vector, ∈ Float64(n_data)
    - D, distance matrix, ∈ Float64(n_data, n_data)
    - θ, tuning param vec, ∈ Float64(M*L)
    - B, matrix containing ϕ_{m,kl}, ∈ Float64(N, M*L)
    - SKs, vector containing SK ∀m, ∈ Float64(N)
    - Midx, vector containing index of supervised data, ∈ Int(M)
    - Widx, vector containing index of unsupervised data ∈ Int(N)
    - cidx, indexer of k or j, ∈ UnitRange(1:M)
    - klidx, vector containing indexer of block column, ∈ UnitRange(M, 1:L) 
    - αj, vector which contains α_j ∀m, ∈ Float64(N)
    - j, absolute index of j ∈ Midx, Int
output:
    - ΔjK, vector ∀m, ∈ Float64(N) (element of outs vector)
    - VK, VK(w_m) ∀m
"""
function comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, αj, j)
    ΔjK, VK, vk, vj, RK, ϕkl, ϕjl = outs;
    @simd for c ∈ cidx # vectorized op on N vector length s.t. x = [m1, m2, ... N]
        k = Midx[c]
        ϕkl .= B[:,klidx[c]]*θ[klidx[c]]
        @. vk = E[k] + ϕkl
        @. RK = RK + (vk/D[Widx, c])
        if j == k # for j term
            ϕjl .= ϕkl
        end
    end
    @. VK = RK / SKs
    @. vj = E[j] + ϕjl
    @. ΔjK = (VK - vj) / αj
end


"""
full ΔjK computer ∀jm, m × j matrix
outputs:
    - vmat, matrix ΔjK(w_m) ∀m,j ∈ Float64(N, M) (preallocated outside!)
    - VK, vector containing VK(w_m) ∀m
"""
function comp_v!(v, vmat, VK, outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α)
    # initial loop for VK (the first one cfant be parallel):
    jc = cidx[1]
    comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α[:, jc], Midx[jc])
    vmat[:, jc] .= outs[1]
    VK .= outs[2] # this only needs to be computed once
    fill!.(outs, 0.)
    # rest of the loop for ΔjK:
    for jc ∈ cidx[2:end]
        comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α[:, jc], Midx[jc])
        vmat[:, jc] .= outs[1]
        fill!.(outs, 0.)
    end
    v .= vec(transpose(vmat))
end

"""
only for VK(w_m) prediction
"""
function comp_VK!(VK, outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx)
    vk, RK, ϕkl = outs;
    @simd for c ∈ cidx # vectorized op on N vector length s.t. x = [m1, m2, ... N]
        k = Midx[c]
        ϕkl .= B[:,klidx[c]]*θ[klidx[c]]
        @. vk = E[k] + ϕkl
        @. RK = RK + (vk/D[Widx, c])
    end
    @. VK = RK / SKs
end

"""
computes the ΔjK across all m ∈ T (vectorized across m)
"""
function comp_ΔjK!(outs, VK, E, θ, B, klidx, αj, jc, j)
    ΔjK, vj, ϕjl = outs;
    ϕjl .= B[:,klidx[jc]]*θ[klidx[jc]]
    @. vj = E[j] + ϕjl
    @. ΔjK = (VK - vj) / αj
end

"""
computes all of the ΔjK (residuals) given VK for j ∈ K, m ∈ T, indexed by j first then m
"""
function comp_res!(v, vmat, outs, VK, E, θ, B, klidx, Midx, α)
    @simd for jc ∈ eachindex(Midx)
        j = Midx[jc]
        comp_ΔjK!(outs, VK, E, θ, B, klidx, α[:, jc], jc, j)
        vmat[:, jc] .= outs[1]
        fill!.(outs, 0.)
    end
    v .= vec(transpose(vmat))
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

"""
==================================
Kernel Ridge Regression models start here!
"""
function comp_gaussian(x, σ)
    return exp.(-x / (2*σ^2))
end

"""
this is also used for prediction, custom, with σ^2 instead of σ
"""
function comp_gaussian_elem(f1, f2, σ2)
    return exp(-norm(f1 - f2)^2 / (2*σ2) )
end

"""
prediction given the indices of W (eval data)
"""
function predict_KRR(F, θ, Widx, Midx, σ2)
    n_pred = length(Widx)
    E = zeros(n_pred)
    @simd for i ∈ eachindex(Widx)
        l = Widx[i]
        @simd for j ∈ eachindex(Midx)
            k = Midx[j]
            E[i] += (comp_gaussian_elem(F[l,:],F[k,:],σ2)*θ[j])
        end
    end
    return E
end

"""
compute the first sigma^2 of the features by taking the mean of ||w-w'||^2
for all w (s.t. it doesnt double count and excludes diagonal ?)
"""
function get_sigma2(F)
    c = 0
    ∑ = 0.
    for i ∈ axes(F, 1)
        for j ∈ axes(F, 1)
            #if i < j
            ∑ += (norm(F[i,:] - F[j, :])^2)
            c += 1
            #end
        end
    end
    return ∑/c
end



"""
===== Molecular gaussian model =====
"""

"""
more specific σ computation (only T × K order of computations)
"""
function get_norms(F, Tidx, Midx)
    rows = length(Tidx); cols = length(Midx)
    norms = zeros(rows, cols)
    for i ∈ eachindex(Midx)
        k = Midx[i]
        for j ∈ eachindex(Tidx) 
            l = Tidx[j]
            norms[j, i] = norm(F[l,:] - F[k, :])^2
        end
    end
    return norms
end

"""
σ0 = <||f_i - f_j||^2>
"""
function get_sigma0(Norms)
    return sum(Norms)/length(Norms)
end

"""
compute gaussian kernel given norm matrix
"""
function comp_gaussian_kernel!(Norms, σ2)
    Norms .= exp.(-Norms/(2*σ2))
end

"""
=== end of molgauss model ===
"""

"""
=== Atomic gaussian model ===
"""

"""
compute the sum of gaussian of atomic features given F_l and F_k
"""
function norms_at(Fk, Fl)
    nk = size(Fk, 1); nl = size(Fl, 1)
    y = zeros(nk*nl)
    c = 1
    for i ∈ axes(Fk, 1)
        for j ∈ axes(Fl, 1)
            y[c] = norm((@view Fk[i,:]) - @view(Fl[j, :]))^2
            c += 1
        end
    end
    return y
end


"""
compute the atomic norms, store it in a Matrix of Vectors: 
    Matrix{Vector{Float64}} where length(Vector) = n_atom^l, size(Matrix) = |Tidx| × |Midx|
"""
function get_norms_at(f, Tidx, Midx)
    rows = length(Tidx); cols = length(Midx)
    norms = Matrix{Vector{Float64}}(undef, rows, cols)
    for i ∈ eachindex(Midx)
        k = Midx[i]
        for j ∈ eachindex(Tidx)
            l = Tidx[j]
            norms[j, i] = norms_at(f[k], f[l])
        end
    end
    return norms
end

"""
get the σ0 out of the atomic features, with input Matrix{Vector{Float64}}
there are 2 versions:
    - mean of total sum 
    - mean of mean (currently used)
"""
function get_sigma0_at(Norms)
    #= s = 0.; n = 0
    for i ∈ eachindex(Norms)
        s += sum(Norms[i])
        n += length(Norms[i])
    end
    return s/n =#
    return mean(map(a -> mean(a), Norms)) # second ver
end

function comp_gaussian_kernel_at(Norms, σ2)
    K = zeros(size(Norms, 1), size(Norms, 2))
    for i ∈ eachindex(Norms)
        K[i] = mean(exp.(-Norms[i]/(2*σ2))) # sum of gaussian of atom
    end
    return K
end
"""
=== end of Atomic gaussian model ===
"""

"""
UNUSED! only for comparison of correctness
generate a gaussian kernel given feature matrix
"""
function comp_gaussian_kernel(F, σ2)
    s = size(F, 1)
    K = zeros(s, s) # symmetric matrix
    for i ∈ axes(F, 1)
        for j ∈ axes(F, 1)
            if j ≥ i
                K[j, i] = comp_gaussian_elem(F[j,:], F[i,:], σ2)
            else
                K[j, i] = K[i, j]
            end
        end
    end
    return K
end

"""
=====================
FCHL-ish kernels
"""

"""
compute the gaussian difference
vector or matrix u,v, scalar cσ := 2*σ^2
"""
function comp_gauss_atom(u, v, cσ)
    return exp((-norm(u - v)^2)/cσ)
end

"""
atomic feature matrix fn of mol n, 
list of atoms ln of mol n, 
gaussian scaler scalar σ 
"""
function comp_atomic_gaussian_entry(f1, f2, l1, l2, cσ)
    entry = 0.
    for i ∈ eachindex(l1)
        for j ∈ eachindex(l2)
            if l1[i] == l2[j] # manually set Kronecker delta using if 
                d = comp_gauss_atom(f1[i, :], f2[j, :], cσ) # (vector, vector, scalar)
                #println(i," ", j, l1[i], l2[j], " ",d)
                entry += d
            end
        end
    end
    return entry
end

"""
return a matrix (length(F1), length(F2)),
params:
    - Fn: vector of atomic features of several molecules
    - Ln: vector of list of atoms
"""
function get_gaussian_kernel(F1, F2, L1, L2, cσ; threading=true)
    nm1 = length(L1); nm2 = length(L2)
    A = zeros(nm1, nm2)
    if threading
        @threads for j ∈ eachindex(L2) # col
            @threads for i ∈ eachindex(L1) # row
                @inbounds A[i, j] = comp_atomic_gaussian_entry(F1[i], F2[j], L1[i], L2[j], cσ)
            end
        end
    else
        for j ∈ eachindex(L2) # col
            for i ∈ eachindex(L1) # row
                A[i, j] = comp_atomic_gaussian_entry(F1[i], F2[j], L1[i], L2[j], cσ)
            end
        end
    end
    return A
end

"""
inner product kernel entry (reproducing kernel)
"""
function comp_repker_entry(u, v)
    return u'v
end

"""
compute repker given 2 feature matrices
"""
function comp_repker(f1, f2)
    K = zeros(size(f1, 1), size(f2, 1))
    @threads for i ∈ axes(f2, 1)
        @threads for j ∈ axes(f1, 1)
            @inbounds K[j,i] = f1[j, :]'f2[i,:]
        end
    end
    return K
end

"""
atomic level repker: K_ll' = ∑_{ij} δ_{il,jl'} K(ϕ_il, ϕ_jl')
similar to gaussian kernel entry
"""
function comp_atomic_repker_entry(f1, f2, l1, l2)
    entry = 0.
    @threads for i ∈ eachindex(l1)
        @threads for j ∈ eachindex(l2)
            @inbounds begin
                if l1[i] == l2[j] # manually set Kronecker delta using if 
                    d = comp_repker_entry(f1[i, :], f2[j, :]) # (vector, vector, scalar)
                    entry += d
                end 
            end
        end
    end
    return entry
end

function get_repker_atom(F1, F2, L1, L2)
    nm1 = length(L1); nm2 = length(L2)
    A = zeros(nm1, nm2)
    @threads for j ∈ eachindex(L2) # col
        @threads for i ∈ eachindex(L1) # row
            @inbounds A[i, j] = comp_atomic_repker_entry(F1[i], F2[j], L1[i], L2[j])
        end
    end
    return A
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
    # flattened feature*basis:
    #S = extract_bspline(x, M; flatten=true)

    # spline using scalar mode, see if the result is the same (and test with AD):
    S, dϕ = extract_bspline_df(x, M)
    display(dϕ)
    for i ∈ [1]#1:n_finger
        pf = plot(vec(x[i,:]), S[i, :, :], legend=false)
        pdf = plot(vec(x[i,:]), dϕ[i, :, :], legend=false)
        savefig(pf, "plot/f.png")
        savefig(pdf, "plot/df.png")
        display(pf)
        display(pdf)
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
    n_data = 10; n_feature = 3; n_basis = 2
    bas = Vector{Float64}(1:n_data)
    W = zeros(n_feature, n_data)
    for i ∈ 1:n_feature
        W[i, :] = bas .+ (0.5*(i-1))
    end
    E = convert(Vector{Float64}, vec(1:5)) # dummy data matrix and energy vector
    println("W = ")
    display(W)
    # dummy distance:
    D = zeros(Float64, n_data, n_data)
    v = 0.
    for j ∈ axes(D, 2)
        for i ∈ axes(D, 1)
            if i == j
                D[i, j] = 0.
                v += 1.
            else
                D[i, j] = v
            end
        end
    end
    D = (D .+ D')./2
    println("D = ")

    Tidx = [1,3,5,7,8,9] # set with length of N, this is from the farthest points heuristics
    Midx = Tidx[[1,3]] # K labels, the known energies
    Uidx = setdiff(Tidx, Midx) # T\K labels, the unsupervised data
    D = D[:, Tidx] # to mimic the new indexing
    display(D)
    data_idx = 1:n_data; Widx = setdiff(data_idx, Midx) # set with length Nqm9-M,for MAE evaluation
    cols = length(Midx)*n_feature*n_basis # index of k,l
    rows = length(Midx)*length(Tidx) # index of j,m  
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

    # tests for precomputing the ϕkl:
    θ = Vector{Float64}(1:cols) # dummy theta
    M = length(Midx); N = length(Widx); L = n_feature*n_basis
    mc = 2; jc = 1; kc = 1
    m = Widx[mc]; j = Midx[jc]; k = Midx[kc]; l = 1
    SKs = map(m -> comp_SK(D, Midx, m), Widx) # precompute vector of SK ∈ R^N for each set of K
    println(["SK", SKs])
    B = zeros(N, M*L)
    comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature) # the index should be k,l only
    println("B=")
    display(B)
    klidx = kl_indexer(M, L) # this is correct, this is the kl indexer!!
    γ = comp_γ(D, SKs, Midx, Widx)
    display(["gamma", γ])
    α = γ .- 1 # precompute alpha matrix for each jm
    outs = [zeros(N) for _ = 1:7];
    cidx = 1:M # k indexer
    v = zeros(N*M); vmat = zeros(N, M); VK = zeros(N); outs = [zeros(N) for _ = 1:7]; 
    comp_v!(v, vmat, VK, outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α) 
    println("residuals:")
    display(vmat)
    display(v)
    #display(VK)
    VKnew = zeros(N); outs = [zeros(N) for _ = 1:3]
    comp_VK!(VKnew, outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx)
    println("VK:")
    display(VKnew)
    v = zeros(N*M); vmat = zeros(N, M); outs = [zeros(N) for _ = 1:3]
    println("new residuals:")
    comp_res!(v, vmat, outs, VKnew, E, θ, B, klidx, Midx, α)
    display(v)
    #ReverseDiff.jacobian(θ->comp_v_j(E, D, θ, B, SKs, Midx, Widx, klidx, j), θ) # for AD, use each jm index and loop it instead of taking the jacobian (slow)

    # test Ax and b routines:
    cidx = 1:M
    temps = [zeros(N) for _ in 1:3]; 
    Ax = zeros(N*M); Axtemp = zeros(N, M) #temporarily put as m × j matrix, flatten later
    comp_Ax!(Ax,Axtemp, temps, θ, B, Midx, cidx, klidx, γ, α)
    #println("Ax:")
    #display(A*θ)
    #display(transpose(Ax)[:]) # default flatten (without transpose) is m index first then j
    temps = [zeros(N) for _ in 1:2]; 
    bnny = zeros(N*M); bnnytemp = zeros(N, M)
    comp_b!(bnny, bnnytemp, temps, E, γ, α, Midx, cidx)
    #display(b)
    #display(transpose(bnny)[:])
    println("Ax - b")
    display(Ax - bnny)
    # test Ax-b comparison:
    #println("norm of (new func - old correct fun) (if 0. then new = correct) = ",norm((A*θ - b) - (transpose(Ax)[:]-transpose(bnny)[:])))

    # test Aᵀv:
    v = rand(N*M); fill!(v, 0.1) # try rand after
    #display(A'*v)
    Aᵀv = zeros(M*L)
    comp_Aᵀv!(Aᵀv, v, B, Midx, Widx, γ, α, L)
    display(Aᵀv)
    #display(norm(Aᵀv - A'*v))
end

"""
test the timing of v vs Aθ - b
"""
function testtime()
    # setup data:
    n_data = 1000; n_feature = 28; n_basis = 8
    W = rand(n_feature, n_data)
    E = rand(n_data)
    # setup data selection:
    M = 100; N = n_data - M
    dataidx = 1:n_data
    Midx = sample(dataidx, M, replace=false)
    Widx = setdiff(dataidx, Midx)
    # compute D, S and ϕ:
    t_data = @elapsed begin
        Bhat = Matrix{Float64}(I, n_feature, n_feature)
        D = compute_distance_all(W, Bhat)
        D = D[:, Midx] # this is just a dummy, this is actually obtained from Eldar's algo
        SKs = map(m -> comp_SK(D, Midx, m), Widx)
        γ = comp_γ(D, SKs, Midx, Widx)
        α = γ .- 1
        ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true)
        n_basis += 3; L = n_feature*n_basis # reset L
        B = zeros(N, M*L); comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature);
        # indexers:
        klidx = kl_indexer(M, L)
        cidx = 1:M
    end
    # residual directly:
    θ = rand(L*M)
    t_v = @elapsed begin
        VK = zeros(N); outs = [zeros(N) for _ = 1:3]
        comp_VK!(VK, outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx)
        v = zeros(N*M); vmat = zeros(N, M); fill!.(outs, 0.)
        comp_res!(v, vmat, outs, VK, E, θ, B, klidx, Midx, α)
    end
    # Ax, b, then residual:
    temps = [zeros(N) for _ in 1:3];
    Ax = zeros(N*M); Axtemp = zeros(N, M) #temporarily put as m × j matrix, flatten later
    t_ax = @elapsed begin
        comp_Ax!(Ax, Axtemp, temps, θ, B, Midx, cidx, klidx, γ, α)
    end
    temps = [zeros(N) for _ in 1:2];
    b = zeros(N*M); btemp = zeros(N, M)
    t_b = @elapsed begin
        comp_b!(b, btemp, temps, E, γ, α, Midx, cidx)
    end
    t_axb = @elapsed begin
        r = Ax - b
    end
    # Aᵀv, try compare norm against actual A too:
    u = rand(N*M)
    Aᵀv = zeros(M*L)
    t_atv = @elapsed begin
        comp_Aᵀv!(Aᵀv, u, B, Midx, Widx, γ, α, L)
    end
    mems = [Base.summarysize(b), Base.summarysize(D), Base.summarysize(SKs), Base.summarysize(ϕ), Base.summarysize(dϕ)].*1e-6 # get storages
    println(mems)       
    println("[t_data, t_res, t_ax, t_b, t_axb, t_atv]", [t_data, t_v, t_ax, t_b, t_axb, t_atv])                                                                                                                                                                                     
    println("M = $M, N = $n_data, L = $n_feature × $n_basis = $L")
    println("norm ax-b - residual = ", norm(r - v))                                                                                                                                                                                                                 
    println("t_atv/t_res = ",t_atv/t_v)
end

"""
testtime using actual data (without data setup time)
complete_batch means the batching starts from the data (hence no intermediate values)
"""
function testtimeactual(foldername, bsize; complete_batch=false)
    # data setup:
    path = "data/$foldername/"
    if complete_batch
        return nothing
    else
        file_dataset = path*"dataset.jld"
        file_finger = path*"features.jld"
        file_distance = path*"distances.jld"
        file_centers = path*"center_ids.jld"
        file_spline = path*"spline.jld"
        file_dspline = path*"dspline.jld"
        files = [file_dataset, file_finger, file_distance, file_centers, file_spline, file_dspline]
        dataset, F, D, Tidx, ϕ, dϕ = [load(f)["data"] for f in files]
    end
    F = F' # always transpose
    E = map(d -> d["energy"], dataset)
    # compute indices:
    n_data = length(dataset); n_feature = size(F, 1);
    Midx = Tidx[1:100]
    Uidx = setdiff(Tidx, Midx) # (U)nsupervised data
    Widx = setdiff(1:n_data, Midx) # for evaluation
    N = length(Tidx); nU = length(Uidx); nK = length(Midx); Nqm9 = length(Widx)
    nL = size(ϕ, 1); n_basis = nL/n_feature
    # all of the vars below depend on Midx, Uidx, Widx:
    t_data = @elapsed begin
        # indexers:
        klidx = kl_indexer(nK, nL)
        cidx = 1:nK
        # intermediate value:
        SKs_train = map(m -> comp_SK(D, Midx, m), Uidx) # only for training, disjoint index from pred
        γ = comp_γ(D, SKs_train, Midx, Uidx)
        SKs = map(m -> comp_SK(D, Midx, m), Widx) # for prediction
        α = γ .- 1
        B = zeros(nU, nK*nL); comp_B!(B, ϕ, dϕ, F, Midx, Uidx, nL, n_feature);
    end
    # fobj comp:
    θ = rand(nK*nL)
    temps = [zeros(nU) for _ in 1:3];
    Ax = zeros(nU*nK); Axtemp = zeros(nU, nK) #temporarily put as m × j matrix, flatten later
    t_ax = @elapsed begin
        comp_Ax!(Ax, Axtemp, temps, θ, B, Midx, cidx, klidx, γ, α)
    end
    temps = [zeros(nU) for _ in 1:2];
    b = zeros(nU*nK); btemp = zeros(nU, nK)
    t_b = @elapsed begin
        comp_b!(b, btemp, temps, E, γ, α, Midx, cidx)
    end
    t_axb = @elapsed begin
        r = Ax - b
    end
    u = rand(nU*nK)
    Aᵀv = zeros(nK*nL)
    t_atv = @elapsed begin
        comp_Aᵀv!(Aᵀv, u, B, Midx, Uidx, γ, α, nL)
    end
    # !!! batch mode prediction!!!:
    # compute batch index:
    if complete_batch
        return nothing
    else
        blength = Nqm9 ÷ bsize # number of batch iterations
        batches = kl_indexer(blength, bsize)
        bend = batches[end][end]
        bendsize = Nqm9 - (blength*bsize)
        push!(batches, bend+1 : bend + bendsize)
        # compute predictions:
        t_batch = @elapsed begin
            VK_fin = zeros(Nqm9)
            B = zeros(Float64, bsize, nK*nL)
            VK = zeros(bsize); outs = [zeros(bsize) for _ = 1:3]
            @simd for batch in batches[1:end-1]
                comp_B!(B, ϕ, dϕ, F, Midx, Widx[batch], nL, n_feature)
                comp_VK!(VK, outs, E, D, θ, B, SKs[batch], Midx, Widx[batch], cidx, klidx)
                VK_fin[batch] .= VK
                # reset:
                fill!(B, 0.); fill!(VK, 0.); fill!.(outs, 0.); 
            end
            # remainder part:
            B = zeros(Float64, bendsize, nK*nL)
            VK = zeros(bendsize); outs = [zeros(bendsize) for _ = 1:3]
            comp_B!(B, ϕ, dϕ, F, Midx, Widx[batches[end]], nL, n_feature)
            comp_VK!(VK, outs, E, D, θ, B, SKs[batches[end]], Midx, Widx[batches[end]], cidx, klidx)
            VK_fin[batches[end]] .= VK
            VK = VK_fin # swap
        end
    end
    # prints :
    println("[Nqm9, N, nK, nf, ns, nL] = ", [Nqm9, N, nK, n_feature, n_basis, nL])
    println("linop timings [t_ax, t_atv] = ", [t_ax, t_atv], ", total = ",sum([t_ax, t_atv]))
    println("timings of [intermediate values, batchpred of VK(w_m) ∀m ∈ Nqm9] = ", [t_data, t_batch])
end



"""
tests LS without forming A (try Krylov.jl and Optim.jl)
"""
function test_LS()
    # try dummy system:
    # setup data:
    n_data = 250; n_feature = 24; n_basis = 6
    W = rand(n_feature, n_data)
    E = rand(n_data)
    # setup data selection:
    M = 100; N = n_data - M
    dataidx = 1:n_data
    Midx = sample(dataidx, M, replace=false)
    Widx = setdiff(dataidx, Midx)
    # compute D, S and ϕ:
    Bhat = Matrix{Float64}(I, n_feature, n_feature)
    D = compute_distance_all(W, Bhat)
    SKs = map(m -> comp_SK(D, Midx, m), Widx)
    γ = comp_γ(D, SKs, Midx, Widx)
    α = γ .- 1
    ϕ, dϕ = extract_bspline_df(W, n_basis; flatten=true, sparsemat=true)
    n_basis += 3; L = n_feature*n_basis # reset L
    B = zeros(N, M*L); comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature);
    # indexers:
    klidx = kl_indexer(M, L)
    cidx = 1:M
    # Ax, b, then residual:
    Axtemp = zeros(N, M); tempsA = [zeros(N) for _ in 1:3]
    # Aᵀv, try compare norm against actual A too:
    v = rand(N*M)
    Aᵀv = zeros(M*L)
    comp_Aᵀv!(Aᵀv, v, B, Midx, Widx, γ, α, L)
    row = N*M; col = L*M
    # Au and Aᵀv as LinearOperator datastructure:
    op = LinearOperator(Float64, row, col, false, false, (y,u) -> comp_Ax!(y, Axtemp, tempsA, u, B, Midx, cidx, klidx, γ, α), 
                                                        (y,v) -> comp_Aᵀv!(y, v, B, Midx, Widx, γ, α, L))
    show(op)
    # compute b:
    b = zeros(N*M); btemp = zeros(N, M); tempsb = [zeros(N) for _ in 1:2]
    t_lo = @elapsed begin
        comp_b!(b, btemp, tempsb, E, γ, α, Midx, cidx)
        #x, stat = cgls(op, b, itmax=500, verbose=1)
    end
    #= display(stat)
    display(norm(op*x - b))
    # compare with standard A, b:
    t_ls = @elapsed begin
        A, b = assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
        x, stat = cgls(A, b, itmax=500)
    end
    display(norm(A*x - b))
    display([t_lo, t_ls]) =#

    # try timer callback:
    function time_callback(solver, start_time, duration)
        return time()-start_time ≥ duration
    end
    start = time()
    x, stat = cgls(op, b, itmax=500, verbose=1, callback=CglsSolver -> time_callback(CglsSolver, start, 2))
    display(stat)
    display(norm(op*x - b))

end


function testrepker()
    
    # === tests on 20--16 features ===
    f = load("data/exp_reduced_energy/features_atom.jld", "data")
    F = load("data/exp_reduced_energy/features.jld", "data") # should try using the curent best found features
    E = readdlm("data/energies.txt")
    dataset = load("data/qm9_dataset_old.jld", "data")
    Eatom = readdlm("data/atomic_energies.txt")
    Ered = E - Eatom
    centers = readdlm("data/centers.txt")[38, 3:102]
    testids = setdiff(1:size(F, 1), centers)
    Ftrain = F[centers,:] #F[centers,:]
    Ftest = F[testids,:]
    
    # test repker as fitter kernel:
    #= K = comp_repker(Ftrain, Ftrain)
    θ, stat = cgls(K, Ered[centers], itmax=500, verbose=0) #θ = K\Ered[centers]
    Epred = K*θ + Eatom[centers]
    display(mean(abs.(Epred - E[centers]))*627.503)
    K = comp_repker(Ftest, Ftrain)
    Epred = K*θ + Eatom[testids]
    display(mean(abs.(Epred - E[testids]))*627.503) =#
    # compare w/ gaussian atom kerneL:
    c = 2048.
    K = get_gaussian_kernel(f[centers], f[centers], [d["atoms"] for d in dataset[centers]], [d["atoms"] for d in dataset[centers]], c)
    display(K)
    θ, stat = cgls(K, Ered[centers], itmax=500, verbose=0) #θ = K\Ered[centers]
    display(mean(abs.(K*θ + Eatom[centers] - E[centers]))*627.503)
    K = get_gaussian_kernel(f[testids], f[centers], [d["atoms"] for d in dataset[testids]], [d["atoms"] for d in dataset[centers]], c)
    display(mean(abs.(K*θ + Eatom[testids] - E[testids]))*627.503)
    
    # test repker as feature:
    #= F = comp_repker(F, Ftrain) # could choose any data points as col
    display(F)
    # get KRR:
    # train:
    σ2 = 2048.
    K = get_norms(F, centers, centers)
    comp_gaussian_kernel!(K, σ2) # generate the kernel
    display(K)
    θ, stat = cgls(K, Ered[centers], itmax=500)
    display(mean(abs.(K*θ + Eatom[centers] - E[centers]))*627.503)
    # test:
    K = get_norms(F, testids, centers)
    comp_gaussian_kernel!(K, σ2)
    display(mean(abs.(K*θ + Eatom[testids] - E[testids]))*627.503) =#

    # test using actual features:
    #= f = load("data/ACSF.jld", "data")
    E = readdlm("data/energies.txt")
    dataset = load("data/qm9_dataset_old.jld", "data")
    Eatom = readdlm("data/atomic_energies.txt")
    Ered = E - Eatom
    centers = readdlm("data/centers.txt")[38, 3:102]
    testids = setdiff(eachindex(dataset), centers) =#

    # test repker atom level:
    K = get_repker_atom(f[centers], f[centers], [d["atoms"] for d ∈ dataset[centers]], [d["atoms"] for d ∈ dataset[centers]])
    display(K)
    θ, stat = cgls(K, Ered[centers], itmax=500) #θ = K\Ered[centers]
    display(mean(abs.(K*θ + Eatom[centers] - E[centers]))*627.503)
    K = get_repker_atom(f[testids], f[centers], [d["atoms"] for d ∈ dataset[testids]], [d["atoms"] for d ∈ dataset[centers]])
    display(mean(abs.(K*θ + Eatom[testids] - E[testids]))*627.503)
end


"""
message passing scheme unit tests:
"""

"""
Message-passing feature transformaation, takes in the whole batch of the dataset (a set of molecuels)
featuring: 
    - asymetric aggregation ⟹ directed graph
    - PCA of the whole dataset instead of one by one foreach t ∈ T
    - uniform n_select ∀t (later would probably be changed to a vector of n_select)

* PCA_params contains the optional parameters of PCA_atom, which is a kwargs dict
"""
function mp_transform(H, T, n_select; PCA_params=Dict())
    println("MP transform starts!")
    e = nothing # init empty e
    tmp = @elapsed begin
        for t ∈ 1:T
            println("timestep = ",t)
            if t == 1
                H, e = mp_step(H, n_select; PCA_params=PCA_params)
            else
                H, e = mp_step(H, n_select; e=e, PCA_params=PCA_params) # now e has already been computed
            end
            println("timestep ",t," is finished!")
        end
    end
    println("MP transform is finished in ",tmp)
    return H
end

"""
MP for one step of t
takes in H the whole molecular dataset, and optional param e the edge features
"""
function mp_step(H, n_select; e=[], PCA_params=Dict())
    # initialization phase:
    nf = size(H[1], 2); nf2 = 2*nf+1 # init feature sizes
    if isempty(e) # check if e is empty ⟹ not yet computed
        e = Vector{Dict}(undef, size(H, 1))
        @threads for l ∈ eachindex(H)
            @inbounds e[l] = mp_getedgef(H[l])
        end
    end
    # aggregation phase 1, concat and sum:
    @threads for l ∈ eachindex(H)
        @inbounds H[l] = mp_agg(H[l], e[l], nf2)
    end
    # aggregation pahse 2, PCA:
    H = PCA_atom(H, n_select; PCA_params...)
    return H, e
end

"""
computes the edge features given node features:
compute interatomic distances given a molecule
the graph is always asumed as full graph (no broken bridges), unless with cutoff
"""
function mp_getedgef(h)
    r = Dict() # store at dict, faster and more efficient, since matmul isnt needed
    @simd for w ∈ axes(h, 1)
        @simd for v ∈ axes(h, 1)
            @inbounds begin
                if w > v # upper triangular
                    r[v,w] = norm(h[v,:] - h[w,:])
                end
            end 
        end
    end
    return r
end

"""
one step of MP aggregate ONLY, given a set of atomic features (one molecule)
nf2 = 2nf + 1 currently
Mt is assumed to be asymmetric, only the distances are ⟹ Mt_vw != Mt_wv ∩ e_vw = e_wv
"""
function mp_agg(h, e, nf2)
    nnodes = size(h, 1) # numofatom
    mt = zeros(nnodes, nf2) # empty vectors of size relevant to the aggregation function, now it is 2nf+1
    @threads for v ∈ axes(h, 1) # loop nodes first
        @threads for w ∈ axes(h, 1) # its "neighbours"
            @inbounds begin
                if v != w # no diag, asymmetry is assumed, can't be neougbours to itself:
                    # only the distances are symmetric:
                    if !haskey(e, (v,w))
                        d = e[w,v]
                    else
                        d = e[v,w]
                    end
                    # mtv = ∑Mt(hv,hw,evw)
                    Mt = vcat(h[v,:], h[w,:], d)
                    mt[v,:] += Mt
                end
            end 
        end
    end
    mt /= (nnodes - 1) # mean
    return mt
end

function testmsg()
    # === tests on 20--16 features ===
    #= f = load("data/exp_reduced_energy/features_atom.jld", "data")
    F = load("data/exp_reduced_energy/features.jld", "data") # should try using the curent best found features
    E = readdlm("data/energies.txt")
    dataset = load("data/qm9_dataset_old.jld", "data")
    Eatom = readdlm("data/atomic_energies.txt")
    Ered = E - Eatom
    centers = readdlm("data/centers.txt")[38, 3:102]
    testids = setdiff(1:size(F, 1), centers)
    Ftrain = F[centers,:] #F[centers,:]
    Ftest = F[testids,:] =#

    # MP test with dummy data:
    #= H = [Matrix{Float64}(I, 4, 3), Matrix{Float64}(2I, 2, 3)] # init dummy data
    T = 5; n_select = 2 # hyperparameters
    pp = Dict() # PCA optional params
    pp[:normalize] = false
    H = mp_transform(H,T,n_select; PCA_params = pp)
    display(H) =#

    # MP test with actual data for fitting:
    # load data:
    f = load("data/exp_reduced_energy/features_atom.jld", "data")
    E = readdlm("data/energies.txt")
    dataset = load("data/qm9_dataset_old.jld", "data")
    Eatom = readdlm("data/atomic_energies.txt")
    Ered = E - Eatom
    centers = readdlm("data/centers.txt")[38, 3:102]
    testids = setdiff(1:length(dataset), centers)
    # mp transform:
    T = 1; n_select = 20
    pp = Dict() # PCA optional params
    f = mp_transform(f,T,n_select; PCA_params = pp)
    # test using repker (current best fitter):
    K = get_repker_atom(f[centers], f[centers], [d["atoms"] for d ∈ dataset[centers]], [d["atoms"] for d ∈ dataset[centers]])
    θ, stat = cgls(K, Ered[centers], itmax=500) #θ = K\Ered[centers]
    display(mean(abs.(K*θ + Eatom[centers] - E[centers]))*627.503) #err train
    K = get_repker_atom(f[testids], f[centers], [d["atoms"] for d ∈ dataset[testids]], [d["atoms"] for d ∈ dataset[centers]])
    display(mean(abs.(K*θ + Eatom[testids] - E[testids]))*627.503) #err test
end


