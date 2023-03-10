

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% paramcheck.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check whether the parameters need to be transformed or not
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%           1       2   3           4           5               6               7       8       9
%naf, nmf in percentage, e.g., .5 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
function tr=paramcheck(typ, lb, ub)
    tr = false;
    dif = abs(ub - lb) <= 100;
    div = abs(ub/(lb+1e-10)) <= 100; % add extra miniscule term for numstability
    if (typ == 1 || dif || div) % integer = 1, real = 0
        tr = false;
    else
        tr = true;
    end
