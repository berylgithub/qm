include("RoSemi.jl")
using Random, DelimitedFiles

function get_repker_atom_old(F1, F2, L1, L2; threading=true)
    if threading
        Fiter = Iterators.product(F1, F2)
        Liter = Iterators.product(L1, L2)
        A = ThreadsX.map((f, l) -> comp_atomic_repker_entry(f[1], f[2], l[1], l[2]),
            Fiter, Liter)
    else
        nm1 = length(L1); nm2 = length(L2)
        A = zeros(nm1, nm2)
        @simd for j ∈ eachindex(L2) # col
            @simd for i ∈ eachindex(L1) # row
                @inbounds A[i, j] = comp_atomic_repker_entry(F1[i], F2[j], L1[i], L2[j])
            end
        end
    end
    return A
end


function test_warm_up()
    strrow = strcol = [["C", "C", "C"],["C", "C", "C"]]
    Random.seed!(603)
    f = [rand(3,3) for i ∈ 1:2]
    get_repker_atom(f[1:2], f[1:2], strrow, strcol)
    c = 2048.
    get_gaussian_kernel(f[1:2], f[1:2], strrow, strcol, c)
    println("warm up done!")
end

function test_actual()
    Random.seed!(603)
    dataset = load("data/qm9_dataset.jld", "data")
    f = load("data/ACSF_51.jld", "data")
    idrows = 1:1000
    idcols = 1:100
    frow = f[idrows]; fcol = f[idcols]; atomsrows = [d["atoms"] for d ∈ dataset[idrows]]; atomscols = [d["atoms"] for d ∈ dataset[idcols]];
    println("DPK:")
    @time K1 = get_repker_atom(frow, fcol, atomsrows, atomscols)

    println("GK:")
    c = 2048.
    @time K1 = get_gaussian_kernel(frow, fcol, atomsrows, atomscols, c)
end

function test_kernels()
    test_warm_up()
    test_actual()
end

function test_corr()
    n_ids = [100, 150, 500]
    println("correctness test")
    Random.seed!(603)
    E = readdlm("data/energies.txt")
    dataset = load("data/qm9_dataset.jld", "data")
    numrow = length(E)
    main_kernels_warmup()
    idall = 1:numrow
    idtest = sample(idall, n_ids[end], replace=false)
    idrem = setdiff(idall, idtest) # remainder ids
    max_n = maximum(n_ids[1:end-1]) # largest ntrain
    max_idtrains = sample(idrem, max_n, replace=false)
    idtrainss = map(n_id -> max_idtrains[1:n_id], n_ids[1:end-1]) # vector of vectors
    f = load("data/FCHL19.jld", "data")
    Fds = [load("data/atomref_features.jld", "data")]
    atomsrow = [d["atoms"] for d ∈ dataset]
    atomscol = [d["atoms"] for d ∈ dataset[max_idtrains]]
    Kr = get_repker_atom(f, f[max_idtrains], atomsrow, atomscol)
    ET = E
    idtrain = idtrainss[1]
    println(idtrain)
    θ = Fds[1][idtrain, :]\ET[idtrain];
    Ea = Fds[1]*θ
    MAEtrain = mean(abs.(ET[idtrain] - Ea[idtrain]))*627.503
    MAEtest = mean(abs.(ET[idtest] - Ea[idtest]))*627.503
    println([MAEtrain, MAEtest])
    ET -= Ea
    trids = indexin(idtrain, max_idtrains)
    println("new kernel code:")
    K = Kr
    Ktr = K[idtrain, trids]
    Ktr[diagind(Ktr)] .+= 1e-8
    θ = Ktr\ET[idtrain]
    Epred = Ktr*θ
    MAEtrain = mean(abs.(ET[idtrain] - Epred))*627.503
    # pred:
    Kts = K[idtest, trids]
    Epred = Kts*θ
    MAEtest = mean(abs.(ET[idtest] - Epred))*627.503
    println([MAEtrain, MAEtest])
    # use the old iterators:
    println("old kernel code:")
    Krold = get_repker_atom_old(f, f[max_idtrains], atomsrow, atomscol)
    K = Krold
    Ktr = K[idtrain, trids]
    Ktr[diagind(Ktr)] .+= 1e-8
    θ = Ktr\ET[idtrain]
    Epred = Ktr*θ
    MAEtrain = mean(abs.(ET[idtrain] - Epred))*627.503
    # pred:
    Kts = K[idtest, trids]
    Epred = Kts*θ
    MAEtest = mean(abs.(ET[idtest] - Epred))*627.503
    println([MAEtrain, MAEtest])
end