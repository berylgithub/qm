%par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
%par_fit_atom = [center_ids] # center_ids = 0 → use 
%par_fit = [model, cσ]
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%           1     2        3           4           5               6             7       8     9
%naf, nmf in percentage, e.g., 50 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
x = [.5, .5, 3, 1, 0, 0, 0, 6, 32.0]' %  example
feas=paramcheck(x)
path_fun = '../data/hyperparamopt/fun.txt';
path_fun = '../test.txt'
newdata = textread(path_fun, "%s")
dumfid = "98safh98afh9"
if (dir(path_fun).bytes > 0) && (~strcmp(newdata{1,1}, dumfid)) % check if file is empty
    disp("newdata found")
else
    disp("newfile is empty or same with prev")
end
