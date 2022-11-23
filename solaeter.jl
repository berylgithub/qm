include("alouEt.jl")

function caller()
    #= data_setup("exp_all_1", 1:133728, 4, 26, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 5, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 6, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    # with sums of squares:
    data_setup("exp_all_1", 1:133728, 4, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 5, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 6, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    # with sums of squares and binomial:
    data_setup("exp_all_1", 1:133728, 4, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=true); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 5, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=true); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 6, 28, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=true); fit_🌹("exp_all_1", 10_000, 900); =#

    data_setup("exp_all_1", 1:133728, 4, 20, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 4, 20, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 4, 26, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 6, 25, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 6, 32, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 5, 36, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    data_setup("exp_all_1", 1:133728, 5, 50, 2, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=true); fit_🌹("exp_all_1", 10_000, 900);

    println("fit experiments complete!!")
end
