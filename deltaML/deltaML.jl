using Base.Threads, DelimitedFiles, LinearAlgebra, Random, StatsBase, Statistics, Krylov 

function gaussian_kernel(A, B, σ)
    # gaussian kernel of two matrices, B determines the number of columns
    K = zeros(size(A, 1), size(B, 1))
    @threads for j ∈ axes(B, 1)
        @threads for i ∈ axes(A, 1)
            @inbounds K[i,j] = norm(A[i,:] - B[j,:])^2
        end
    end
    K = exp.( -K ./ (2*σ^2))
    return K
end

function laplacian_kernel(A, B, σ)
    # laplacian kernel of two matrices, B determines the number of columns
    K = zeros(size(A, 1), size(B, 1))
    @threads for j ∈ axes(B, 1)
        @threads for i ∈ axes(A, 1)
            @inbounds K[i,j] = norm(A[i,:] - B[j,:], 1)
        end
    end
    K = exp.( -K ./ σ)
    return K
end

function getDistances(R)
    # computes interatomic distances given coordinates
    # R row := number of atoms
    natom = size(R, 1)
    D = zeros(natom, natom)
    ids = 1:natom
    for c ∈ Iterators.product(ids, ids)
        D[c[1],c[2]] = norm(R[c[1],:]-R[c[2],:])
    end
    return D
end

function sumpairpot(Z, D; α = 1., δ = 1.) # hyperparameters for later use
    # sum of pair potentials given list of atom nuclear charges and interatomi distances (the atomic order must be the same)
    # the output is symm-like, hence we only take the upper triangular to avoid double count
    natom = length(Z)
    ids = 1:natom
    E = 0. #zeros(natom, natom)
    for c ∈ Iterators.product(ids, ids)
        if c[1] < c[2] # only count upper triangular
            E += Z[c[1]]*Z[c[2]]/(D[c[1], c[2]])^δ
        end
    end
    return E
end


function compare_fit()
    # main obj: fit using julia's CGLS, compare result with cho_solve

    # load X:
    X = readdlm("deltaML/data/qm7coulomb.txt")

    # load E:
    E = readdlm("deltaML/data/hof_qm7.txt")
    E_hof = vec(float.(E[:, 2])); E_dftb = vec(float.(E[:, 3])); E_delta = E_hof-E_dftb
    
    # load ids
    basepath = "deltaML/data/train_indexes_"
    ndata = length(E_hof); rangedata = range(1,ndata)
    idtrains = [vec(Int.(readdlm(basepath*"1000.txt"))).+1, vec(Int.(readdlm(basepath*"2000.txt"))).+1, vec(Int.(readdlm(basepath*"4000.txt"))).+1]
    idtests = [setdiff(rangedata, ids) for ids ∈ idtrains]
    
    display(idtrains[1])
    # fit
    
    σ = 700.
    for i ∈ eachindex(idtrains)
        # Etot:
        Xtrain = X[idtrains[i], :]; Xtest = X[idtests[i], :]
        Ytrain = E_hof[idtrains[i]]; Ytest = E_hof[idtests[i]] 
        K = gaussian_kernel(Xtrain, Xtrain, σ)
        #α, stat = cgls(K, Ytrain, itmax = 500, λ = 1e-8)
        K[diagind(K)] .+= 1e-8
        α = K\Ytrain
        K = gaussian_kernel(Xtest, Xtrain, σ)
        Ypred = K*α
        MAEtot = mean(abs.(Ypred - Ytest))
        # Edelta: 
        Xtrain = X[idtrains[i], :]; Xtest = X[idtests[i], :]
        Ytrain = E_delta[idtrains[i]]; Ytest = E_delta[idtests[i]] 
        K = gaussian_kernel(Xtrain, Xtrain, σ)
        #α, stat = cgls(K, Ytrain, itmax = 500, λ = 1e-8)
        K[diagind(K)] .+= 1e-8
        α = K\Ytrain
        K = gaussian_kernel(Xtest, Xtrain, σ)
        Ypred = K*α
        MAEdelta = mean(abs.(Ypred - Ytest))
        println("(Ntrain, MAEtot, MAEdelta) = ",[length(idtrains[i]), MAEtot, MAEdelta])
    end
end

function fit_zaspel()
    #= 
    zaspel either did (perhaps i try both):
     - directly use energy as base
     - multilevel training, the bases are also trained
    some infos in the plot (fig 5):
     - the target is always the highest basis+corr combo: CCSDT+ccpvfdz for any plots
     - upper left: 2DCQML with fixed basis set: ccpvdz, but varying the corr
     - upper right: 2D fixed corr=ccsdt, 3D varying corr+base+data obviously 
     - bottom: MAE and RMSE
    other info:
     ? $s$ as the ratio of dataset is still unclear
     ? probably the indices of has a pattern if the sum of indices = max index, then it's +, otherwise -, (for β)
    for zaspel guess i can start by doing both direct and multitrain using the end basis+corr as target while taking any intermediates as bases -> turns out this doesnt use the target energy (?)
    =#
    # load data:
    datapath = "C:/Users/beryl/OneDrive/Dokumente/Dataset/zaspel_supp/"
    X = readdlm(datapath*"features_coulomb_zaspel.txt")
    l_basis = ["E_sto3g", "E_631g", "E_ccpvdz"] # ordered from the cheapest
    l_corr = ["E_hf", "E_mp2", "E_ccsdt"] # same
    E = Dict() # nested dict of energy, inner = corr, outer = base
    for basis ∈ l_basis
        etemp = readdlm(datapath*basis*".txt")
        E[basis] = Dict()
        for (i,corr) ∈ enumerate(l_corr)
            E[basis][corr] = etemp[:, i]
        end 
    end
    
    # define EΔ and Etarget
    EΔ = E["E_ccpvdz"]["E_ccsdt"] - E["E_sto3g"]["E_hf"] # change here manually
    Etarget = E["E_ccpvdz"]["E_ccsdt"]

    # shuffled index for training
    Random.seed!(603)
    ndata = size(X, 1); nrange = range(1, ndata)
    trainsize = [100, 1000, 4000]
    idtrains = [sample(1:ndata, siz, replace=false) for siz ∈ trainsize]
    idtests = [setdiff(nrange, ids) for ids ∈ idtrains]

    # fit target
    σ = 400.
    for i ∈ eachindex(idtrains)
        # Etarget fitting:
        Xtrain = X[idtrains[i], :]; Xtest = X[idtests[i], :]
        Ytrain = Etarget[idtrains[i]]; Ytest = Etarget[idtests[i]] 
        K = laplacian_kernel(Xtrain, Xtrain, σ)
        K[diagind(K)] .+= 1e-8
        α = K\Ytrain
        K = laplacian_kernel(Xtest, Xtrain, σ)
        Ypred = K*α
        MAEtot = mean(abs.(Ypred - Ytest))
        
        # EΔ fitting:
        Ytrain = EΔ[idtrains[i]]; Ytest = EΔ[idtests[i]] 
        K = laplacian_kernel(Xtrain, Xtrain, σ)
        K[diagind(K)] .+= 1e-8
        α = K\Ytrain
        K = laplacian_kernel(Xtest, Xtrain, σ)
        Ypred = K*α
        MAEΔ = mean(abs.(Ypred - Ytest))
        
        println("(Ntrain, MAEtot, MAEΔ) = ",[length(idtrains[i]), MAEtot, MAEΔ])

        # ΔML:

    end

end

