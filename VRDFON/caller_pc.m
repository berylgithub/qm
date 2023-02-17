%par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
%par_fit_atom = [center_ids] # center_ids = 0 → use 
%par_fit = [model, cσ]
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%naf, nmf in percentage, e.g., 50 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
x = [.5, .5, 3, -1, 0, 0, 0, 6, 32.0]'; %  example
feas=paramcheck(x);
prevbest = textread('../data/best_fun_params.txt', "%s")
f = str2double(prevbest{1,1})
x = zeros(length(prevbest)-1,1)
for i=2:length(prevbest)
    x(i-1) = str2double(prevbest{i,1});
end
if exist('../data/best_fun_params.txsdt', 'file')
    disp("exists")
end