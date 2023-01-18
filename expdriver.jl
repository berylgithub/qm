include("alouEt.jl")

function caller()
    data_setup("exp_reduced_energy", 1:133728, 95, 22, 3, 300, "data/qm9_dataset_old.jld", "data/SOAP.jld", "SOAP")
    models = ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
    for model ∈ models
        fit_🌹_and_atom("exp_reduced_energy", 10_000, 900; model = model)
    end
end
