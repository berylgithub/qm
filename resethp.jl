# resets all hyperparameter optimization outputs (for linux)

paths = ["data/hyperparamopt/fun.txt",
        "data/hyperparamopt/params.txt",
        "data/hyperparamopt/raw_params.txt",
        "data/hyperparamopt/flist.txt",
        "data/hyperparamopt/xlist.txt",
        "data/hyperparamopt/xrawlist.txt",
        "data/hyperparamopt/tracker.txt",
        "data/hyperparamopt/sim/sim_tracker.txt"]

for pt ∈ paths
    run(`rm $pt`)
    run(`touch $pt`)
end
run(`rm -r data/hyperparamopt/sim/f/*`)
run(`rm -r data/hyperparamopt/sim/x/*`)