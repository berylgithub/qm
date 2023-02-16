% check wether the params are feasible, if infeasible, return very high fobjvalue
%par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
%par_fit_atom = [center_ids] # center_ids = 0 → use 
%par_fit = [model, cσ]
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%           1       2   3           4           5               6               7       8       9
%naf, nmf in percentage, e.g., .5 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
function feas=paramcheck(x)
feas=true; % return fail if the constraints are violated) 

% check negatives:
for i=1:length(x)
    if x(i) < 0
        feas=false;
        break
        return
    end
end

% check datasetup params: 
if x(1) == 0.0 || x(2) == 0.0 || x(1) > 1.0 || x(2) > 1.0 
    feas=false;
    return 
end
if x(3) == 0 
    feas=false;
    return
end
if x(4) == 0 || x(4) > 3 
    feas=false;
    return
end
if x(5) > 1 || x(6) > 1
    feas=false;
    return
end

% model params:
if x(8) > 5
    f=false;
    return
end
if x(9) == 0
    f=false;
    return
end


