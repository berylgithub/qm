using Krylov
using DelimitedFiles
using Statistics
using LinearAlgebra

function gaussian_kernel(A, B, σ)
    # gaussian kernel of two matrices, B is columnwise
    K = zeros(size(A, 1), size(B, 1))
    @simd for j ∈ axes(B, 1)
        @simd for i ∈ axes(A, 1)
            @inbounds K[i,j] = norm(A[i,:] - B[j,:])^2
        end
    end
    K = exp.( -K ./ (2*σ^2))
    return K
end

function compare_fit()
    # !!! missing the kernels!! print it to file using the py code, or just write, simple
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

compare_fit()