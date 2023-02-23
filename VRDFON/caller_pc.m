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
path_fun = '../test.txt';
path_bounds = '../data/hyperparamopt/bounds.txt';
bounds = dlmread(path_bounds)
xlo = bounds(1,2); xhi = bounds(2,2);
xtest = -2.
xtest = max(xlo, min(xtest, xhi)) %clip to boundary
isinf(bounds(2,2))

xc = 0.8;
xl=floor(xc);
xh=ceil(xc);
f = xc-xl
1-f

% appending to matrix (row wise) and finding vector in matrix (row wise)
A = [[1.111,2.235,3];[4,5,6]];
A = [A;[-1,-1,-1]]
b = [1.1111,2.235,3];
rid = find(ismember(A, b,'rows'))
dlmwrite(path_fun, b, "\t")
dlmread(path_fun)
path_trackx = '../data/hyperparamopt/xlist.txt';
exist(path_trackx)

xlist = [[1.111 2.235 3];[4 5 6]]
flist = [1,2];
x = [1.11111,2.235,3]'
f = 4;
[f, xlist, flist] = paramtracker(x, f, xlist, flist)
