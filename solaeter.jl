include("alouEt.jl")

function caller()
    #data_setup("exp_all_1", 1:133728, 4, 18, 2, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900); # warm start
    nmf = 20:26; n_bas = 5:8
    for i ∈ nmf
        for j ∈ n_bas
           if i <= 23 && j <= 8
	      continue
	   else
              data_setup("exp_all_1", 1:133728, 4, i, j, 300, "data/ACSF_atom.jld"; ft_sos=false, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
    	   end
        end
    end
    nmf = 21:30; n_bas = 2:5
    for i ∈ nmf
        for j ∈ n_bas
	   if i <= 23 && j <= 6
	      continue
	   else
              data_setup("exp_all_1", 1:133728, 4, i, j, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=false); fit_🌹("exp_all_1", 10_000, 900);
	   end
        end
    end
    for i ∈ nmf
        for j ∈ n_bas
            data_setup("exp_all_1", 1:133728, 4, i, j, 300, "data/ACSF_atom.jld"; ft_sos=true, ft_bin=true); fit_🌹("exp_all_1", 5_000, 900);
        end
    end
    println("fit experiments complete!!")
end
