include("RoSemi.jl")
using Random, DelimitedFiles

"""
inner product kernel entry (reproducing kernel)
"""
function comp_repker_entry_t(u::AbstractArray, v::AbstractArray)::Float64
    return u'v
end

"""
atomic level repker: K_ll' = ∑_{ij} δ_{il,jl'} K(ϕ_il, ϕ_jl')
similar to gaussian kernel entry
"""
function comp_atomic_repker_entry_t(f1::AbstractArray, f2::AbstractArray, l1::Vector{String}, l2::Vector{String})::Float64
    entry = 0.
    @simd for i ∈ eachindex(l1)
        @simd for j ∈ eachindex(l2)
            @inbounds begin
                if l1[i] == l2[j] # manually set Kronecker delta using if 
                    @views d = comp_repker_entry_t(f1[i, :], f2[j, :]) # (vector, vector, scalar)
                    entry += d
                end 
            end
        end
    end
    return entry
end

function get_repker_atom_t(F1::AbstractArray, F2::AbstractArray, L1::Vector{Vector{String}}, L2::Vector{Vector{String}}; threading=true)::Matrix{Float64}
    if threading
        rowids = eachindex(L1); colids = eachindex(L2)
        iterids = Iterators.product(rowids, colids)
        A = ThreadsX.map((tuple_id) -> comp_atomic_repker_entry_t(F1[tuple_id[1]], F2[tuple_id[2]], L1[tuple_id[1]], L2[tuple_id[2]]), iterids)
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

function test_dt()
    strrow = strcol = [["C", "C", "C"],["C", "C", "C"]]
    Random.seed!(603)
    f = [rand(3,3)' for i ∈ 1:2]
    K = get_repker_atom(f[1:2], f[1:2], strrow, strcol)
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
    ET = E
    idtrain = idtrainss[1]
    # try using the pre-generated ids:
    idss = load("data/test_corr_ids.jld", "data")
    idtrain = idss[1]; idtest = idss[3]; max_idtrains = idss[2]
    atomscol = [d["atoms"] for d ∈ dataset[max_idtrains]]
    Kr = get_repker_atom(f, f[max_idtrains], atomsrow, atomscol)
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
end