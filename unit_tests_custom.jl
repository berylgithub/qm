#include("RoSemi.jl")
using Random

function test_warm_up()
    strrow = strcol = [["C", "C", "C"],["C", "C", "C"]]
    Random.seed!(603)
    f = [for i ∈ 1:2 rand(3,3)]
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