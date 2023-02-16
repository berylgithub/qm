%par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
%par_fit_atom = [center_ids] # center_ids = 0 → use 
%par_fit = [model, cσ]
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%naf, nmf in percentage, e.g., 50 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
x = [.5, .5, 3, -1, 0, 0, 0, 6, 32.0]' %  example
feas=paramcheck(x)
p = rand(1)
if p < 0.5
    y=floor(.5)
else
    y=ceil(.5)
end
