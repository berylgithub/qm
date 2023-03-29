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




x = [3, 1, 0, 0]; ct = 1
xprev(1:2) = x(1:2)
while true
    % chekc break counter
    if ct >= 5
        disp("reset at")
        disp(ct)
        ct = 1 % reset count
        break
    end
    % get new x:
    x(3:4) = rand(1,2)
    disp(ct)
    disp("")
    % compare if new x is equal to prev x
    if x(1:2) == xprev(1:2)
        ct += 1
    end
    % set new prev:
    xprev(3:4) = x(3:4)
end

x = [-1.234; 29823.345; 0.899435]
sum(minprob(x))
