# resets all hyperparameter optimization outputs (for linux)

paths = ["data/hyperparamopt/fun.txt",
        "data/hyperparamopt/params.txt",
        "data/hyperparamopt/raw_params.txt",
        "data/hyperparamopt/flist.txt",
        "data/hyperparamopt/xlist.txt",
        "data/hyperparamopt/xrawlist.txt",
        "data/hyperparamopt/tracker.txt",
        "data/hyperparamopt/sim/sim_tracker.txt"]
# recreate files:
for pt ∈ paths
    rm(pt)
    io = open(pt, "w")
    close(io)
end
# remove only:
files = vcat(readdir("data/hyperparamopt/sim/f/"), readdir("data/hyperparamopt/sim/x/"))
for f ∈ files
    rm(f)
end