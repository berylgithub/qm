include("RoSemi.jl")
using Random


function comp_gauss_atom_v2(u::Vector{Float64}, v::Vector{Float64}, c::Float64)::Float64
    return exp((-norm(u - v)^2)/c)
end

function comp_atomic_gaussian_entry_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, l1::Vector{String}, l2::Vector{String}, c::Float64)::Float64
    entry = 0.
    @simd for i ∈ eachindex(l1)
        @simd for j ∈ eachindex(l2)
            @inbounds begin
                if l1[i] == l2[j] # manually set Kronecker delta using if 
                    d = comp_gauss_atom_v2(f1[i, :], f2[j, :], c) # (vector, vector, scalar)
                    entry += d
                end 
            end
        end
    end
    return entry
end

function get_gaussian_kernel_v2(F1::Vector{Matrix{Float64}}, F2::Vector{Matrix{Float64}}, L1::Vector{Vector{String}}, L2::Vector{Vector{String}}, c::Float64; threading=true)::Matrix{Float64}
    if threading
        Fiter = Iterators.product(F1, F2)
        Liter = Iterators.product(L1, L2)
        A = ThreadsX.map((f, l) -> comp_atomic_gaussian_entry_v2(f[1], f[2], l[1], l[2], c),
            Fiter, Liter)
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


function comp_repker_entry_v2(u::Vector{Float64}, v::Vector{Float64})::Float64
    return u'v
end

function comp_atomic_repker_entry_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, l1::Vector{String}, l2::Vector{String})::Float64
    entry = 0.
    @simd for i ∈ eachindex(l1)
        @simd for j ∈ eachindex(l2)
            @inbounds begin
                if l1[i] == l2[j] # manually set Kronecker delta using if 
                    d = comp_repker_entry_v2(f1[i, :], f2[j, :]) # (vector, vector, scalar)
                    entry += d
                end 
            end
        end
    end
    return entry
end


function get_repker_atom_v2(F1::Vector{Matrix{Float64}}, F2::Vector{Matrix{Float64}}, L1::Vector{Vector{String}}, L2::Vector{Vector{String}}; threading=true)::Matrix{Float64}
    if threading
        Fiter = Iterators.product(F1, F2)
        Liter = Iterators.product(L1, L2)
        A = ThreadsX.map((f, l) -> comp_atomic_repker_entry_v2(f[1], f[2], l[1], l[2]),
            Fiter, Liter)
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



function test_kernels()
    Random.seed!(603)
    dataset = load("data/qm9_dataset.jld", "data")
    f = load("data/ACSF_51.jld", "data")
    idrows = 1:200
    idcols = 1:100
    # warm up:
    println("warm_up")
    get_repker_atom_v2(f[1:2], f[1:2], [d["atoms"] for d ∈ dataset[1:2]], [d["atoms"] for d ∈ dataset[1:2]])
    get_repker_atom(f[1:2], f[1:2], [d["atoms"] for d ∈ dataset[1:2]], [d["atoms"] for d ∈ dataset[1:2]])
    # test direct vs of dpk:
    println("with view")
    frow = @view f[idrows]; fcol = @view f[idcols]; atomsrows = [d["atoms"] for d ∈ dataset[idrows]]; atomscols = [d["atoms"] for d ∈ dataset[idcols]];
    println("with vartypes:")
    @time K1 = get_repker_atom_v2(frow, fcol, atomsrows, atomscols)
    println("without:")
    @time K2 = get_repker_atom(frow, fcol, atomsrows, atomscols)
    display(K1)
    display(K2)
end