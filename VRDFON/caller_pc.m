%par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
%par_fit_atom = [center_ids] # center_ids = 0 → use 
%par_fit = [model, cσ]
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%           1     2        3           4           5               6             7       8     9
%naf, nmf in percentage, e.g., 50 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
x = [.5, .5, 3, 1, 0, 0, 0, -6, -2048.0]' %  example
path_fun = '../data/hyperparamopt/fun.txt';
path_fun = '../test.txt';
path_bounds = '../data/hyperparamopt/bounds.txt';
bounds = dlmread(path_bounds)

%{
i=8;
lb = bounds(1,i)/(1+abs(bounds(1,i))); ub = bounds(2,i)/(1+abs(bounds(2,i)));
bs = [ub, lb]
bs = sort(bs)
z = max(bs(1), min(x(i), bs(2)))
z = z/(1-z)
%}

xdiff = abs(x);
for i=1:length(x)
    xdiff(i) = xdiff(i)/bounds(2, i)
end
sum(xdiff)

100/(0+1e-10)
100/100