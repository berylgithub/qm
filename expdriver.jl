include("alouEt.jl")

function caller()
    models = ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
    for model ∈ models
        fit_🌹_and_atom("exp_reduced_energy", 10_000, 900; model = model)
    end
end
