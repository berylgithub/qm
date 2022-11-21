using LinearAlgebra, Statistics, Distributions, Plots, LaTeXStrings

"""
placeholder for linear algebra and statistics operations, if RoSemi is overcrowded, or when the need arises
"""

"""
the Kronecker delta function
"""
δ(x,y) = Float64(==(x,y))

"""
determine the memory size (in megabytes) of A matrix given M, N, L
"""
function compmem(M, N, L)
    return M^2*N*L*64/8e6
end


"""
computes residual from data matrix A, coefficient vector θ, and target vector b
"""
function residual(A, θ, b)
    return A*θ - b
end

"""
return the Least squares form (norm^2) of the residual r := Aθ - b
"""
function lsq(A, θ, b)
    return norm(residual(A, θ, b))^2
end

"""
compute mean vector and covariance matrix
params:
    - idx, the index of the wanted molecule (w0)
    - N, number of data
    - len_finger, vector length of the fingerprint
output:
    - wbar, mean, size = len_finger
    - C, covariance, size = (len_finger, len_finger)
"""
function mean_cov(w_matrix, idx, N, len_finger)
    # initialize intermediate vars:
    S = zeros(len_finger,len_finger)
    dw = zeros(len_finger)
    dif = zeros(len_finger, N)

    loopset = union(1:idx-1, idx+1:N) # loop index
    # dw := 
    for i ∈ loopset # all except the w0 index itself, since w0 - w0 = 0
        dif[:,i] = w_matrix[:, i] .- w_matrix[:, idx] #wv- w0
        dw .+= dif[:, i]
    end
    dw ./= N
    # S := 
    for i ∈ loopset
        S .+= dif[:,i]*dif[:,i]'
    end

    #= display(w_matrix[:, idx])
    display(dif)
    display(dw)
    display(S)
    display(dw*dw') =#

    return w_matrix[:, idx] .+ dw, (S .- (dw*dw'))./(N-1) # mean, covariance
end

"""
eigenvalue distribution plotter
personal use only
"""
function plot_ev(v, tickslice, filename; rotate=false)
    # put small egeinvalues to zero:
    b = abs.(v) .< 1e-9
    v[b] .= 0.
    v_nz = [v_el for v_el in v if v_el > 0.]
    idx_v = [i for i in eachindex(v) if v[i] > 0.]
    v_nz = reverse(v_nz); idx_v = reverse(idx_v)
    v_rev = reverse(v)
    # compute distribution:
    display(v)
    display(v_nz)
    display(idx_v)
    #= for j=1:len
        q[j] = sum(v[end-j+1:end])
    end
    q ./= trace
    q = reverse(q) =#


    # plot:
    #p = scatter(log10.(q), xticks = (eachindex(q)[tickslice], (eachindex(q).-1)[tickslice]), markershape = :cross, xlabel = L"$j$", ylabel = L"log$_{10}$($q_{j}$)", legend = false)
    #slicer = Int.(round.(collect(range(1, length(idx_v), 20))))
    #display(slicer)
    #p = scatter(log10.(v_nz), xticks = (eachindex(v_nz)[slicer], eachindex(v_nz)[slicer]), markershape = :cross, xlabel = L"$i$", ylabel = L"log$_{10}$($\lambda_{i}$)", legend = false, xrotation = -45, xtickfontsize=6)
    p = scatter(log10.(v_rev), xticks = (eachindex(v_rev)[tickslice], eachindex(v_rev)[tickslice]), markershape = :cross, xlabel = L"$i$", ylabel = L"log$_{10}$($\lambda_{i}$)", legend = false, xtickfontsize=6)
    display(p)
    savefig(p, filename)
end

"""
feature selection by the PCA
params:
    - W, the full data matrix (133k × n_feature)
    - n_select, the number of features to be selected
output:
    - U, the full data matrix but with n_select columns
"""
function PCA(W, n_select)
    n_mol, n_f = size(W)
    s = vec(sum(W, dims=1)) # sum over all molecules
    # long ver, more accurate, memory safe, slower:
    #= S = zeros(n_f, n_f)
    for i ∈ 1:n_mol
        S .+= W[i,:]*W[i,:]'
    end =#
    S = W'*W # short ver, careful of memory alloc!
    u_bar = s./n_mol
    C = S/n_mol - u_bar*u_bar'
    display(C)
    e = eigen(C)
    v = e.values
    Q = e.vectors
    #display(norm(C - Q*diagm(v)*Q')) # this is correct, small norm
    sidx = sortperm(v, rev=true) # sort by largest eigenvalue
    # descend sort the v and Q (by column, by definition!!):
    v = v[sidx]
    #println([sum(v), sum(v[1:n_select]), sum(v[1:n_select])/sum(v)])
    Q = Q[:, sidx] # according to Julia factorization: F.vectors[:, k] is the kth eigenvector
    #display(norm(C-Q*diagm(v)*Q'))
    # >>>> for plotting purpose only!!, comment when not needed!: <<<<
    #= p = plot(sidx, log.(v), xlabel = L"$i$", ylabel = L"log($\Lambda_{ii}$)", legend = false)
    display(p)
    savefig(p, "plot/eigenvalues.png") =#
    # compute the ratio of the dropped eigenvalues/trace (v is already ordered descending):
    #= trace = sum(v)
    len = length(v)
    q = zeros(len)
    for j=1:len
        q[j] = sum(v[end-j+1:end])
    end
    q ./= trace
    q = reverse(q)
    tickslice = [1,20,40,60, 80, 102]
    p = scatter(log10.(q), xticks = (eachindex(q)[tickslice], (eachindex(q).-1)[tickslice]), markershape = :cross, xlabel = L"$j$", ylabel = L"log$_{10}$($q_{j}$)", legend = false)
    display(p)
    savefig(p, "plot/ev_dropped_ratio.png") =#
    # >>>> end of plot <<<<
    # select the n_select amount of number of features:
    v = v[1:n_select]
    Q = Q[:, 1:n_select]
    #display(norm(C-Q*diagm(v)*Q'))
    U = zeros(n_mol, n_select)
    for i ∈ 1:n_mol 
        U[i, :] = Q'*(W[i, :] - u_bar) #Q'*(u_bar - W[i, :]) # Q^T*(u - ̄u) !!
    end
    return U
end

"""
PCA for atomic features
params:
    - atomic features, f ∈ (N,n_atom,n_f)
    ;
    - callplot is for personal use only
"""
function PCA_atom(f, n_select; normalize=true, callplot=false)
    # cut number of features:
    N, n_f = (length(f), size(f[1], 2))
    # compute mean vector:
    s = zeros(n_f); ∑ = zeros(n_f)
    @simd for l ∈ 1:N
        n_atom = size(f[l], 1)
        @simd for i ∈ 1:n_atom
            ∑ .= ∑ .+ f[l][i,:] 
        end
        ∑ .= ∑ ./ n_atom
        s .= s .+ ∑
        fill!(∑, 0.) # reset
    end
    s ./= N
    # intermediate matrix:
    S = zeros(n_f, n_f); ∑S = zeros(n_f, n_f)
    @simd for l ∈ 1:N
        n_atom = size(f[l], 1)
        @simd for i ∈ 1:n_atom
            ∑S .= ∑S .+ (f[l][i,:]*f[l][i,:]')
        end
        ∑S .= ∑S ./ n_atom
        S .= S .+ ∑S
        fill!(∑S, 0.)
    end
    S ./= N
    # covariance matrix:
    C = S - s*s'
    #C = (S - s*s') ./ (N - 1)
    # spectral decomposition:
    e = eigen(C)
    v = e.values # careful of numerical overflow and errors!!
    Q = e.vectors
    # plot here:
    #plot_ev(v, [1,10,20,30,40,50], "plot/log_eigenvalue_atom.png")
    #= U, sing, V = svd(C) # for comparison if numerical instability ever arise, SVD is more stable
    display(sing) =#
    # check if there exist large negative eigenvalues (most likely from numerical overflow), if there is try include it:
    # sort from largest eigenvalue instead:
    sidx = sortperm(v, rev=true)
    v = v[sidx]
    Q = Q[:, sidx]
    # select eigenvalues:
    v = v[1:n_select]
    #display(v)
    Q = Q[:, 1:n_select]
    #display(norm(C-Q*diagm(v)*Q'))
    f_new = Vector{Matrix{Float64}}(undef, N)
    @simd for l ∈ 1:N
        n_atom = size(f[l], 1)
        temp_A = zeros(n_atom, n_select)
        @simd for i ∈ 1:n_atom
            temp_A[i,:] .= Q'*(f[l][i,:] - s)
        end
        f_new[l] = temp_A
    end
    # normalize
    if normalize
        maxs = map(f_el -> maximum(f_el, dims=1), f_new); maxs = vec(maximum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), maxs)), dims=1))
        mins = map(f_el -> minimum(f_el, dims=1), f_new); mins = vec(minimum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), mins)), dims=1))
        @simd for l ∈ 1:N
            n_atom = size(f[l], 1)
            @simd for i ∈ 1:n_atom
                f_new[l][i,:] .= (f_new[l][i,:] .- mins) ./ (maxs .- mins) 
            end
        end
    end
    return f_new
end

"""
this should be extracted after PCA_atom, hence it's here
params:
    - atomic features, f ∈ (N,n_atom,n_f)
"""
function comp_mol_l!(s, S, fl, n_atom)
    for i ∈ 1:n_atom
        s .= s .+ fl[i, :]
        S .= S .+ (fl[i, :]*fl[i, :]')
    end
end

function extract_mol_features(f)
    N, n_f = (length(f), size(f[1], 2))
    n_mol_f = Int(2*n_f + n_f*(n_f - 1)/2)
    F = zeros(N, n_mol_f)
    s = zeros(n_f)
    S = zeros(n_f, n_f)
    for l ∈ 1:N
        n_atom = size(f[l], 1)
        comp_mol_l!(s, S, f[l], n_atom)
        F[l, :] .= vcat(s, S[triu!(trues(n_f, n_f))])
        fill!(s, 0.); fill!(S, 0.) # reset
    end
    return F
end

"""
this separates each atomic features into block vectors: [H,C,N,O,F, n_x/n, 1/n],
where 1/n is a scalar and x ∈ HCONF
assume features and dataset are contiguous
"""
function extract_mol_features(f, dataset)
    N, n_f0 = (length(f), size(f[1], 2))
    n_f = n_f0*5 # since the features are separated
    #n_mol_f = Int(2*n_f + n_f*(n_f - 1)/2) + 6 # 6 = 5 distinct type + 1 sum 
    types = ["H", "C", "N", "O", "F"]
    F = zeros(N, n_f+6) #zeros(N, n_mol_f)
    # initialize Dict:
    fd = Dict()
    for typ in types
        fd[typ] = zeros(Float64, n_f0)
    end
    fs = Dict()
    for typ in types
        fs[typ] = 0.
    end
    fl = zeros(n_f); S = zeros(n_f)
    # compute feature for each mol:
    for l ∈ eachindex(f) 
        n_atom = dataset[l]["n_atom"]
        atoms = dataset[l]["atoms"]
        # compute features for each atom:
        for i ∈ eachindex(atoms)
            fd[atoms[i]] += f[l][i,:]
            fs[atoms[i]] += 1.0 # count the number of atoms
        end
        # concat manual, cleaner:
        fd_at = vcat(fd["H"], fd["C"], fd["N"], fd["O"], fd["F"])
        fs_at = vcat(fs["H"], fs["C"], fs["N"], fs["O"], fs["F"]) ./ n_atom
        # compute the upper triangular matrix:

        # combine everything:
        F[l,:] = vcat(fd_at, fs_at, 1/n_atom)
        # reset fd:
        for typ in types
            fd[typ] .= 0.
            fs[typ] = 0.
        end
    end
    #display(dataset[100])
    #println(F[100,:],)
    return 
end

"""
PCA for molecule (matrix data type)
params:
    - F, ∈Float64(N, n_f)
"""
function PCA_mol(F, n_select; normalize=true)
    N, n_f = size(F)
    s = zeros(n_f); S = zeros(n_f, n_f)
    comp_mol_l!(s, S, F, N)
    s ./= N; S ./= N
    C = S - s*s'

    e = eigen(C)
    v = e.values # careful of numerical overflow and errors!!
    Q = e.vectors
    #println("ev compute done")
    # plot here:
    #plot_ev(v, round.(range(1, n_f, 20)), "plot/log_eigenvalue_mol.png")

    sidx = sortperm(v, rev=true)
    v = v[sidx] # temporary fix for the negative eigenvalue
    Q = Q[:, sidx] # temporary fix for the negative eigenvalue
    # select eigenvalues:
    v = v[1:n_select]
    Q = Q[:, 1:n_select]

    F_new = zeros(N, n_select)
    for l ∈ 1:N
        F_new[l,:] .= Q'*(F[l,:] - s)
    end

    if normalize
        maxs = vec(maximum(F_new, dims=1))
        mins = vec(minimum(F_new, dims=1))
        for l ∈ 1:N
            F_new[l,:] .= (F_new[l,:] .- mins) ./ (maxs .- mins)
        end
    end
    return F_new
end


"""
caller for PCA_atom up to PCA_mol
"""
function feature_extractor(f, n_select_at, n_select_mol; normalize_at=true, normalize_mol=true)
    f = PCA_atom(f, n_select_at; normalize = normalize_at)
    F = extract_mol_features(f)
    F = PCA_mol(F, n_select_mol; normalize = normalize_mol)
    return F
end

function checkcov(X)
    r, c = size(X)
    s = vec(sum(X, dims=1)) ./ r
    S = zeros(c, c)
    for i ∈ 1:r
        S .= S .+ (X[i, :]*X[i, :]')
    end
    S ./= r
    C = S - s*s'
    #C = (S - s*s') ./ (r - 1)
    return C
end

"""
compute the B linear transformer for Mahalanobis distance
params:
    - C, covariance matrix of the fingerprints
outputs:
    - B, the linear transformer for Mahalanobis distance, used for: ||B(w - wk)||₂²
"""
function compute_B(C)
    # eigendecomposition:
    e = eigen(C)
    #display(e)
    # round near zeros to zero, numerical stability:
    v = e.values
    bounds = [-1e-8, 1e-8]
    b = bounds[1] .< v .< bounds[2]
    v[b] .= 0.
    v = v.^(-.5) # take the "inverse sqrt" of the eigenvalue vector
    Q = e.vectors
    # eigenvalue regularizer:
    dmax = 1e4*minimum(v) # take the multiple minimum of the diagonal
    #display(v)
    v = min.(dmax, v)
    #display(v)
    D = diagm(v)
    # compute B:
    return D*Q' # B = D*Qᵀ
end

function test_cov()
    # const:
    N = 3
    len_finger = 2
    # inputs:
    wmat = [1 2 3; 1 2 3]
    idx = 2 # w0 idx
    # func:
    wbar, C = mean_cov(wmat, idx, N, len_finger)
    display(wbar)
    display(C)
    B = compute_B(C)
    display(B)
    display(B'*B)
end

function normalize_routine(infile)
    W = load(infile)["data"]
    #W = W' # 1 data = 1 column
    dt = StatsBase.fit(UnitRangeTransform, W, dims=1)
    W = StatsBase.transform(dt, W)
    return W
end

"""
overloader, takes the feature matrix instead of file
"""
function normalize_routine(W)
    dt = StatsBase.fit(UnitRangeTransform, W, dims=1)
    W = StatsBase.transform(dt, W)
    return W
end


"""
compute the binomial(m, 2) feature from PCA(W, 51)
params:
    - m, num of selected features after PCA
"""
function extract_binomial_feature(m)
    W_half = load("data/ACSF_symm.jld")["data"][:, 52:102] # only include the sum features
    W_pca = PCA(W_half, m)
    # generate index:
    bin = binomial(m, 2)
    b_ind = zeros(Int, bin, 2)
    c = 1
    for j ∈ 1:m
        for i ∈ 1:m
            if i<j
                b_ind[c,:] .= [i, j]
                c += 1
            end
        end
    end
    # compute feature:
    n_f = m + bin
    W_new = zeros(size(W_half,1), n_f)
    W_new[:, 1:m] .= W_pca 
    for i ∈ eachindex(b_ind[:, 1])
        W_new[:, m+i] .= W_pca[:, b_ind[i, 1]] .* W_pca[:, b_ind[i, 2]]
    end
    save("data/ACSF_PCA_bin_$n_f.jld", "data", W_new)
end