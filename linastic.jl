using LinearAlgebra, Statistics, Distributions, Plots

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
PCA for atomic features 
params:
    - atomic features, f ∈ (N,n_atom,n_f)
"""
function PCA_atom(f, n_select)
    N, n_f = (length(f), size(f[1], 2))
    # compute mean vector:
    s = zeros(n_f)
    for l ∈ 1:N
        n_atom = size(f[l], 1)
        for i ∈ 1:n_atom
            s .= s .+ f[l][i,:] 
        end
    end
    s .= s./N
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