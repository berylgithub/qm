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
a = -2; b = 2; N=3;
xp = a + (b-a) .* rand(N,1)
x = [0; 0; 1] + xp
p = minprob(x)
sum(p)
A = rand(3,5);
size(bounds)(1)
%}
bounds
binfo = extractbound(bounds)
x = rand(binfo(end, end), 1)
xi = x(binfo(4,1):binfo(4,2))
p = minprob(xi)
z = 0;
r = 0.990
q = [0.; cumsum(p)]
for i=1:length(q)-1
    if (q(i) < r) && (r <= q(i+1))
        z = i;
        break
    end
end
z