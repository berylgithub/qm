%par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
%par_fit_atom = [center_ids] # center_ids = 0 → use 
%par_fit = [model, cσ]
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%           1     2        3           4           5               6             7       8     9
%naf, nmf in percentage, e.g., 50 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
%x = [.5, .5, 3, 1, 0, 0, 0, -6, -2048.0]' %  example
path_fun = '../data/hyperparamopt/fun.txt';
path_x = '../data/hyperparamopt/params.txt'
path_bounds = '../data/hyperparamopt/bounds.txt';
bounds = dlmread(path_bounds);

% x = [0.5; 0.5; 6; ]
x = [0.5; 0.5; 5.5; 1/3; 1/3; 1/3; 1/2; 1/2; 1/2; 1/2; 38; 1/5; 1/5; 1/5; 1/5; 1/5; 10.5;] % encoded x
length(x)
bm = extractbound(bounds)
[xnew, fpen] = decode(x, bounds, bm)
