include("expdriver.jl")

# spawns simulator ONE BY ONE to preserve paralelization, based on the available ids


#= id = 1
simfiles = readdir("data/hyperparamopt/sim/f/")
while true
    str = "sim_$id.txt"
    if str ∉ simfiles
        break
    end
    id += 1
end
println("spawned simulator with id = $id !!!")
writedlm("data/hyperparamopt/sim/f/sim_$id.txt", [], "\t")
#hyperparamopt_parallel(id; dummyfx=true, trackx = true)
=#

test_largedata()