include("RoSemi.jl")
using Random


function comp_gauss_atom_v2(u::Union{Vector{Float64}, SubArray{Float64, 1}}, v::Union{Vector{Float64}, SubArray{Float64, 1}}, c::Float64)::Float64
    return exp((-norm(u - v)^2)/c)
end

function comp_atomic_gaussian_entry_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, l1::Vector{String}, l2::Vector{String}, c::Float64)::Float64
    entry = 0.
    @simd for i ∈ eachindex(l1)
        @simd for j ∈ eachindex(l2)
            @inbounds begin
                if l1[i] == l2[j] # manually set Kronecker delta using if 
                    d = comp_gauss_atom_v2(@view(f1[i, :]), @view(f2[j, :]), c) # (vector, vector, scalar)
                    entry += d
                end 
            end
        end
    end
    return entry
end

function get_gaussian_kernel_v2(F1::Vector{Matrix{Float64}}, F2::Vector{Matrix{Float64}}, L1::Vector{Vector{String}}, L2::Vector{Vector{String}}, c::Float64; threading=true)::Matrix{Float64}
    if threading
        rowids = eachindex(L1); colids = eachindex(L2)
        iterids = Iterators.product(rowids, colids)
        A = ThreadsX.map((tuple_id) -> comp_atomic_gaussian_entry_v2(F1[tuple_id[1]], F2[tuple_id[2]], L1[tuple_id[1]], L2[tuple_id[2]], c), iterids)
    else
        nm1 = length(L1); nm2 = length(L2)
        A = zeros(nm1, nm2)
        @simd for j ∈ eachindex(L2) # col
            @simd for i ∈ eachindex(L1) # row
                @inbounds A[i, j] = comp_atomic_gaussian_entry_v2(F1[i], F2[j], L1[i], L2[j], c)
            end
        end
    end
    return A
end


function comp_repker_entry_v2(u::Union{Vector{Float64}, SubArray{Float64, 1}}, v::Union{Vector{Float64}, SubArray{Float64, 1}})::Float64
    return u'v
end

function comp_atomic_repker_entry_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, l1::Vector{String}, l2::Vector{String})::Float64
    entry = 0.
    @simd for i ∈ eachindex(l1)
        @simd for j ∈ eachindex(l2)
            @inbounds begin
                if l1[i] == l2[j] # manually set Kronecker delta using if 
                    @views d = comp_repker_entry_v2(f1[i, :], f2[j, :]) # (vector, vector, scalar)
                    entry += d
                end 
            end
        end
    end
    return entry
end


function get_repker_atom_v2(F1::Vector{Matrix{Float64}}, F2::Vector{Matrix{Float64}}, L1::Vector{Vector{String}}, L2::Vector{Vector{String}}; threading=true)::Matrix{Float64}
    if threading
        rowids = eachindex(L1); colids = eachindex(L2)
        iterids = Iterators.product(rowids, colids)
        A = ThreadsX.map((tuple_id) -> comp_atomic_repker_entry_v2(F1[tuple_id[1]], F2[tuple_id[2]], L1[tuple_id[1]], L2[tuple_id[2]]), iterids)
    else
        nm1 = length(L1); nm2 = length(L2)
        A = zeros(nm1, nm2)
        @simd for j ∈ eachindex(L2) # col
            @simd for i ∈ eachindex(L1) # row
                @inbounds A[i, j] = comp_atomic_repker_entry_v2(F1[i], F2[j], L1[i], L2[j])
            end
        end
    end
    return A
end

function test_warm_up()
    dataset = load("data/qm9_dataset.jld", "data")
    f = load("data/ACSF_51.jld", "data")
    get_repker_atom_v2(f[1:2], f[1:2], [d["atoms"] for d ∈ dataset[1:2]], [d["atoms"] for d ∈ dataset[1:2]])
    get_repker_atom(f[1:2], f[1:2], [d["atoms"] for d ∈ dataset[1:2]], [d["atoms"] for d ∈ dataset[1:2]])
    c = 2048.
    get_gaussian_kernel_v2(f[1:2], f[1:2], [d["atoms"] for d ∈ dataset[1:2]], [d["atoms"] for d ∈ dataset[1:2]], c)
    get_gaussian_kernel(f[1:2], f[1:2], [d["atoms"] for d ∈ dataset[1:2]], [d["atoms"] for d ∈ dataset[1:2]], c)
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
    println("new:")
    @time K1 = get_repker_atom_v2(frow, fcol, atomsrows, atomscols)
    println("old:")
    @time K2 = get_repker_atom(frow, fcol, atomsrows, atomscols)
    display(norm(K1-K2))

    println("GK:")
    c = 2048.
    println("new:")
    @time K1 = get_gaussian_kernel_v2(frow, fcol, atomsrows, atomscols, c)
    println("old:")
    @time K2 = get_gaussian_kernel(frow, fcol, atomsrows, atomscols, c)
    display(norm(K1-K2))
end

function test_kernels()
    #test_warm_up()
    test_actual()
end