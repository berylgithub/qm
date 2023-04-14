using LinearAlgebra, Statistics, StatsBase, Distributions, Plots, LaTeXStrings, DataFrames

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
    v_rev = reverse(v) ./ v[end] # λi/λ1
    #display(v)
    #display(v_rev)
    #display(idx_v)
    # compute distribution:
    μ = sum(v_rev)
    init_μ = μ
    μs = zeros(length(v_rev))
    for i ∈ eachindex(v_rev)
        μs[i] = μ
        μ = max(0., μ - v_rev[i])
    end
    #display(μ)
    #display(μs)
    μs = μs ./ init_μ
    # remove zeros:
    μs_nz = [v_el for v_el in μs if v_el ≥ 1e-10]
    idx_μ = [i for i in eachindex(μs) if μs[i] ≥ 1e-10]
    #display(μs_nz)
    # plots:
    #p = scatter(log10.(v_rev), xticks = (eachindex(v_rev)[tickslice], eachindex(v_rev)[tickslice]), markershape = :cross, xlabel = L"$i$", ylabel = L"$log_{10}(\lambda_{i}/\lambda_1)$", legend = false, xtickfontsize=6);
    #display(p)
    savefig(scatter(log10.(v_rev), xticks = (eachindex(v_rev)[tickslice], eachindex(v_rev)[tickslice]), markershape = :cross, xlabel = L"$i$", ylabel = L"$log_{10}(\lambda_{i}/\lambda_1)$", legend = false, xtickfontsize=6), filename*"_rat.png")

    # replace tickslice here:
    tickslice = Int.(round.(range(1, length(μs_nz), 15)))
    #p = scatter(log10.(μs_nz), xticks = (eachindex(μs_nz)[tickslice], eachindex(μs_nz)[tickslice]), yticks = (-10.0:0.), markershape = :cross, xlabel = L"$i$", ylabel = L"$log_{10}(\mu_{i}/\mu_1)$", legend = false, xtickfontsize=6);
    #display(p)
    savefig(scatter(log10.(μs_nz), xticks = (eachindex(μs_nz)[tickslice], eachindex(μs_nz)[tickslice]), yticks = (-10.0:0.), markershape = :cross, xlabel = L"$i$", ylabel = L"$log_{10}(\mu_{i}/\mu_1)$", legend = false, xtickfontsize=6), filename*"_dist.png")
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
function PCA_atom(f, n_select; normalize=true, normalize_mode="minmax", fname_plot_at="", save_cov=false)
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
    if save_cov
        save("data/covariance_matrix_atomic.jld", "data", C)
    end
    # correlation matrix:
    D = diagm(1. ./ .√ C[diagind(C)])
    C = D*C*D # the diagonals are the sensitivity
    # spectral decomposition:
    e = eigen(C)
    v = e.values # careful of numerical overflow and errors!!
    Q = e.vectors
    #display(v)
    # plot here:
    #plot_ev(v, Int.(round.(LinRange(1, n_f, n_select))), "plot/ev_atom_"*fname_plot_at)
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
    #= f_new = Vector{Matrix{Float64}}(undef, N)
    @simd for l ∈ 1:N
        n_atom = size(f[l], 1)
        temp_A = zeros(n_atom, n_select)
        @simd for i ∈ 1:n_atom
            temp_A[i,:] .= Q'*(f[l][i,:] - s)
        end
        f_new[l] = temp_A
    end =#
    # try memory efficient op, but more risky!:
    @simd for l ∈ 1:N
        n_atom = size(f[l], 1)
        temp_A = zeros(n_atom, n_select)
        @simd for i ∈ 1:n_atom
            temp_A[i,:] .= Q'*(f[l][i,:] - s)
        end
        f[l] = temp_A
    end
    # normalize
    if normalize
        if normalize_mode == "minmax"
            maxs = map(f_el -> maximum(f_el, dims=1), f); maxs = vec(maximum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), maxs)), dims=1))
            mins = map(f_el -> minimum(f_el, dims=1), f); mins = vec(minimum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), mins)), dims=1))
            @simd for l ∈ 1:N
                n_atom = size(f[l], 1)
                @simd for i ∈ 1:n_atom
                    f[l][i,:] .= (f[l][i,:] .- mins) ./ (maxs .- mins) 
                end
            end
        elseif normalize_mode == "ecdf" # empirical CDF scaler, UNFINISHED, DONT USE!
            maxs = map(f_el -> maximum(f_el, dims=1), f); maxs = vec(maximum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), maxs)), dims=1))
            mins = map(f_el -> minimum(f_el, dims=1), f); mins = vec(minimum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), mins)), dims=1))
            @simd for l ∈ 1:N
                n_atom = size(f[l], 1)
                @simd for i ∈ 1:n_atom
                    f[l][i,:] .= (f[l][i,:] .- mins) ./ (maxs .- mins) 
                end
            end    
        end
    end
    s = ∑ = S = ∑S = C = D = e = temp_A = nothing; GC.gc() # clear memory
    return f
end


"""
PCA given covariance matrix C, sensitivity vector σ, and atomic features f
"""
function PCA_atom(f, n_select, C, σ; normalize=true, fname_plot_at="")
    # # correlation matrix:
    #D = diagm(1. ./ .√ C[diagind(C)])
    #σ = C[diagind(C)]
    D = diagm(1. ./ σ) # i think it should be diagm(1/σ), since it's supposed to be divider
    C = D*C*D # the diagonals are the sensitivity
    # spectral decomposition:
    e = eigen(C)
    v = e.values # careful of numerical overflow and errors!!
    Q = e.vectors
    # compute mean of the features:
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
    # plot here:
    #plot_ev(v, Int.(round.(LinRange(1, n_f, n_select))), "plot/ev_atom_"*fname_plot_at)
    # sort from largest eigenvalue instead:
    sidx = sortperm(v, rev=true)
    v = v[sidx]
    Q = Q[:, sidx]
    # select eigenvalues:
    v = v[1:n_select]
    #display(v)
    Q = Q[:, 1:n_select]
    #= f_new = Vector{Matrix{Float64}}(undef, N)
    @simd for l ∈ 1:N
        n_atom = size(f[l], 1)
        temp_A = zeros(n_atom, n_select)
        @simd for i ∈ 1:n_atom
            temp_A[i,:] .= Q'*(f[l][i,:] - s)
        end
        f_new[l] = temp_A
    end =#
    @simd for l ∈ 1:N
        n_atom = size(f[l], 1)
        temp_A = zeros(n_atom, n_select)
        @simd for i ∈ 1:n_atom
            temp_A[i,:] .= Q'*(f[l][i,:] - s)
        end
        f[l] = temp_A
    end
    # normalize
    if normalize
        maxs = map(f_el -> maximum(f_el, dims=1), f); maxs = vec(maximum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), maxs)), dims=1))
        mins = map(f_el -> minimum(f_el, dims=1), f); mins = vec(minimum(mapreduce(permutedims, vcat, map(m_el -> vec(m_el), mins)), dims=1))
        @simd for l ∈ 1:N
            n_atom = size(f[l], 1)
            @simd for i ∈ 1:n_atom
                f[l][i,:] .= (f[l][i,:] .- mins) ./ (maxs .- mins) 
            end
        end
    end
    s = ∑ = C = D = e = temp_A = nothing; GC.gc() # clear memory
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
divider to avoid NaNs
"""
function dividenan(x, a)
    return isnan((x ./ a)[1]) ? zeros(length(x)) : x ./ a   
end

"""
this separates each atomic features into block vectors: [H,C,N,O,F, n_x/n, 1/n],
where 1/n is a scalar and x ∈ HCONF
assume features and dataset are contiguous
params:
    - mode: features to be used: fsos, fbin
"""
function extract_mol_features(f, dataset; ft_sos=false, ft_bin=false, sum_mode=0)
    N, n_f0 = (length(f), size(f[1], 2))
    #n_f = n_f0*5*2 + (binomial(n_f0, 2)+n_f0)*5 # since the features are separated, ×2 since the includes also the quadratic, + binomial(n_f0, 2)*5 since it's the atomic combination
    n_f = n_f0*5
    if ft_sos
        n_f *= 2
    end
    if ft_bin
        n_f += ((binomial(n_f0, 2)+n_f0)*5)
    end
    types = ["H", "C", "N", "O", "F"]
    F = zeros(N, n_f+6) #zeros(N, n_mol_f) #+6 for the tail features (the count of the atoms)
    # initialize Dicts:
    fd = Dict() # sum features
    for typ in types
        fd[typ] = zeros(Float64, n_f0)
    end
    fs = Dict() # the atomic counts
    for typ in types
        fs[typ] = 0.
    end
    if ft_sos
        fds = Dict() # sum of squares features
        for typ in types
            fds[typ] = zeros(Float64, n_f0)
        end
    end
    if ft_bin
        fbin = Dict() # binomial feature matrix for each atom type
        for typ in types
            fbin[typ] = zeros(Float64, n_f0, n_f0)
        end
    end
    # compute feature for each mol:
    for l ∈ eachindex(f) 
        n_atom = dataset[l]["n_atom"]
        atoms = dataset[l]["atoms"]
        # compute features for each atom:
        for i ∈ eachindex(atoms)
            fd[atoms[i]] += f[l][i,:]
            if ft_sos
                fds[atoms[i]] += (f[l][i,:] .^ 2)
            end
            if ft_bin
                fbin[atoms[i]] += (f[l][i,:]*f[l][i,:]') # compute the upper triangular matrix (binomial features) only from the summed ACSF features
            end
            fs[atoms[i]] += 1.0 # count the number of atoms
        end
        # concat manual, cleaner:
        ## sum modes:
        if sum_mode == 0 # standard sum mode
            fd_at = vcat(fd["H"], fd["C"], fd["N"], fd["O"], fd["F"])
        elseif sum_mode == 1 # each atom type average
            fd_at = vcat(dividenan(fd["H"], fs["H"]), dividenan(fd["C"], fs["C"]), dividenan(fd["N"], fs["N"]), dividenan(fd["O"], fs["O"]), dividenan(fd["F"], fs["F"]))
        else # average of whole atoms
            fd_at = vcat(fd["H"], fd["C"], fd["N"], fd["O"], fd["F"]) ./ (fs["H"] + fs["C"] + fs["N"] + fs["O"] + fs["F"])
        end
        # count of atoms:    
        fs_at = vcat(fs["H"], fs["C"], fs["N"], fs["O"], fs["F"]) ./ n_atom # N_X/N
        if ft_sos
            fds_at = vcat(fds["H"], fds["C"], fds["N"], fds["O"], fds["F"])
        end
        if ft_bin
            fbin_at = vcat(fbin["H"][triu!(trues(n_f0, n_f0))], fbin["C"][triu!(trues(n_f0, n_f0))], fbin["N"][triu!(trues(n_f0, n_f0))], fbin["O"][triu!(trues(n_f0, n_f0))], fbin["F"][triu!(trues(n_f0, n_f0))])
        end
        # combine everything:
        z = fd_at
        if ft_sos
            z = vcat(z, fds_at)
        end
        if ft_bin
            z = vcat(z, fbin_at)
        end
        z = vcat(z, fs_at, 1/n_atom)
        F[l,:] = z
        # reset dicts:
        for typ in types
            fd[typ] .= 0.
            fs[typ] = 0.
            if ft_sos
                fds[typ] .= 0.
            end
            if ft_bin
                fbin[typ] .= 0.
            end
        end
    end
    #display(dataset[1])
    #println(f[1][2,:].^2 + f[1][3,:].^2 + f[1][4,:].^2 + f[1][5,:].^2)
    #display(f[1][2,1:5]*f[1][2,1:5]' + f[1][3,1:5]*f[1][3,1:5]' + f[1][4,1:5]*f[1][4,1:5]' + f[1][5,1:5]*f[1][5,1:5]')
    #println(F[1,511:550])
    return F
end


"""
molecular feature ONLY from atomic counts
"""
function get_atom_counts(dataset)
    types = ["H", "C", "N", "O", "F"]
    N = length(dataset)
    F = zeros(length(types), N) # output, transposed
    fs = Dict() # the atomic counts
    for typ in types
        fs[typ] = 0.
    end
    for l ∈ eachindex(dataset)
        # compute feature:
        n_atom = dataset[l]["n_atom"]
        atoms = dataset[l]["atoms"] # vector of list of atoms
        for i ∈ eachindex(atoms)
            fs[atoms[i]] += 1.0 # count the number of atoms
        end
        # fill feature:
        F[:, l] = vcat(fs["H"], fs["C"], fs["N"], fs["O"], fs["F"])
        # reset dicts:
        for typ in types
            fs[typ] = 0.
        end
    end
    return transpose(F) # back to row dominant
end


"""
PCA for molecule (matrix data type)
params:
    - F, ∈Float64(N, n_f)
    - cov test only for numerically unstable features (such as FCHL)
"""
function PCA_mol(F, n_select; normalize=true, normalize_mode = "minmax", cov_test=true, fname_plot_mol="")
    N, n_f = size(F)
    if cov_test
        C = cov(F)
        D = diagm(1. ./ .√ C[diagind(C)])
        fids = findall(c -> (c == Inf)||isnan(c), D)
        # remove features:
        exids = [id[1] for id ∈ fids]
        println("removed mol features:", exids)
        newids = setdiff(1:n_f, exids)
        F_new = zeros(N, length(newids))
        for l ∈ 1:N
            F_new[l, :] = F[l, newids]
        end
        F = F_new
        n_f = size(F, 2)
    end
    s = zeros(n_f); #S = zeros(n_f, n_f)
    #comp_mol_l!(s, S, F, N)
    for i ∈ 1:N
        s .= s .+ F[i, :]
    end
    s ./= N; #S ./= N
    
    #C = S - s*s' # covariance matrix

    # correlation matrix:
    C = cor(F) # more accurate than the D*C*D somehow
    # here should check for Infs or NaNs first
    e = eigen(C)
    v = e.values # careful of numerical overflow and errors!!
    Q = e.vectors
    #display(v)
    #println("ev compute done")
    # plot here:
    #plot_ev(v, Int.(round.(range(1, n_f, n_select))), "plot/ev_mol_"*fname_plot_mol)

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
        if normalize_mode == "minmax"
            maxs = vec(maximum(F_new, dims=1))
            mins = vec(minimum(F_new, dims=1))
            for l ∈ 1:N
                F_new[l,:] .= (F_new[l,:] .- mins) ./ (maxs .- mins)
            end
        elseif normalize_mode == "ecdf" # empirical CDF scaler
            @simd for k ∈ axes(F_new, 2)
                ec = ecdf(F_new[:, k]) # fit CDF
                F_new[:, k] = ec(F_new[:, k]) # predict CDF
            end
        end
    end
    C=e=F=nothing; GC.gc() # clear memory
    return F_new
end


"""
caller for PCA_atom up to PCA_mol
"""
function feature_extractor(f, dataset, n_select_at, n_select_mol; normalize_at=true, normalize_mol=true, ft_sos=true, ft_bin=true, fname_plot_at="", fname_plot_mol="")
    f = PCA_atom(f, n_select_at; normalize = normalize_at, fname_plot_at=fname_plot_at)
    #F = extract_mol_features(f)
    F = extract_mol_features(f, dataset; ft_sos = ft_sos, ft_bin = ft_bin) # the one with separated atomtype
    F = PCA_mol(F, n_select_mol; normalize = normalize_mol, fname_plot_mol=fname_plot_mol)
    return F
end

"""
standalone function to compute empirical cumulative distribution function (ecdf)
"""
function comp_ecdf(f, ids; type = "mol")
    if type == "mol"
        for i ∈ axes(f, 2) # fit ecdf for each feature/column
            f[:, i] = ecdf(f[ids, i])(f[:, i]) # fit ecdf using selected ids and predict ∀ indexes
        end
    end
    return f
end

"""
function to flatten vector of matrices to matrices with the same number of columns
also returns the relative indexing of the matrix
"""
function flattener(f)
    # get the rowsizes of the matrices:
    natoms = []
    for l ∈ eachindex(f)
        push!(natoms, size(f[l],1))
    end
    lbs = []; ubs = []; # vectors containing the flattened matrix indices
    # fill the indexer vectors:
    lb = 0; ub = 0;
    for i ∈ eachindex(natoms)
        if i == 1
            lb = 1
        else
            lb = ub+1
        end
        ub = lb+natoms[i]-1
        push!(lbs, lb); push!(ubs, ub)
    end
    # fill the flattened matrix:
    ff = zeros(sum(natoms), size(f[1], 2)) # doubles memory, careful!!
    for l ∈ axes(f,1)
        ff[lbs[l]:ubs[l], :] = f[l]
    end
    return ff, lbs, ubs
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


"""
compute feature sensitivity given actual f and several perturbed f (fs)
"""
function get_feature_sensitivity(file_f, files_fs)
    f = load(file_f)["data"]
    f_ps = [load(fil)["data"] for fil in files_fs]
    errors = Vector{Vector{Float64}}()
    t = @elapsed begin
        for i ∈ eachindex(f)
            for j ∈ axes(f[i], 1) # for each atom:
                errs = [abs.(f[i][j, :] - f_p[i][j, :]) for f_p in f_ps]
                maxerr = max.(errs...)
                push!(errors, maxerr)
            end
        end
    end
    MAE_mat = reduce(hcat, errors)' # matrix of size N × n_af
    display(MAE_mat)
    println("elapsed = ",t)
    MAEs = mean(MAE_mat, dims=1)
    return vec(MAEs) # vector of length n_f 
end

"""
self use function
"""
function process_FCHL()
    dataset = load("data/qm9_dataset_old.jld", "data")
    f = load("data/FCHL.jld", "data")
    ndata = length(dataset)
    fp = Vector{Array{Float64, 3}}(undef, ndata)
    # drop zeros and placeholders (such as very large number):
    for l ∈ eachindex(dataset)
        natom = dataset[l]["n_atom"]
        fp[l] = f[l, 1:natom, :, 1:natom]
    end
    save("data/FCHL.jld", "data", fp)
end