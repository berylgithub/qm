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
path_bounds_old = '../data/hyperparamopt/bounds_old.txt';
bounds = dlmread(path_bounds);
bounds_old = dlmread(path_bounds_old);

% x = [0.5; 0.5; 6; ]
xold = [0.5; 0.5; 5.5; 1/3; 1/3; 1/3; 1/2; 1/2; 1/2; 1/2; 38; 1/5; 1/5; 1/5; 1/5; 1/5; 10.5;]; % encoded x
f = 100.;

x = [1/3; 1/3; 1/3; 5.5; 1/3; 1/3; 1/3; 1/2; 1/2; 1/2; 1/2; 1/5; 1/5; 1/5; 1/5; 1/5; 10.5]

%{
zi = [1/6, 1/3, 1/3, 1/6]'
xu = 20; xl = 1;
p = minprob(zi)
x = computex(p, xl, xu)
p = computep(x, xl, xu)
norm(p-zi)**2

bm = extractbound(bounds_old)
decode(xold, bounds_old, bm)
%}


bounds
newbounds = boundtransform(bounds)
bm = extractbound(newbounds)
%decode(x, bounds, bm)
